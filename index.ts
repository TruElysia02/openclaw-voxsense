import { randomBytes } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import type {
  OpenClawPluginApi,
  PluginHookBeforePromptBuildResult,
  PluginHookMessageContext,
  PluginHookMessageReceivedEvent,
  PluginHookMessageSendingEvent,
  PluginHookMessageSendingResult,
} from "openclaw/plugin-sdk";
import { SILENT_REPLY_TOKEN, requireApiKey } from "openclaw/plugin-sdk/compat";

type PluginMode = "reply" | "handoff";

type RawPluginConfig = {
  provider?: string;
  model?: string;
  channels?: string[];
  historyMessages?: number;
  maxHistoryChars?: number;
  timeoutMs?: number;
  onlyWhenNoText?: boolean;
  mode?: PluginMode;
  requireStructuredResponse?: boolean;
  suppressDefaultReply?: boolean;
  suppressTranscriptEcho?: boolean;
  storeHeardTextInSession?: boolean;
  debug?: boolean;
  systemPrompt?: string;
  failureNotice?: string;
};

type ResolvedPluginConfig = {
  provider: string;
  model: string;
  channels: string[];
  historyMessages: number;
  maxHistoryChars: number;
  timeoutMs: number;
  onlyWhenNoText: boolean;
  mode: PluginMode;
  requireStructuredResponse: boolean;
  suppressDefaultReply: boolean;
  suppressTranscriptEcho: boolean;
  storeHeardTextInSession: boolean;
  debug: boolean;
  systemPrompt?: string;
  failureNotice?: string;
};

type PluginRuntimeState = {
  active: boolean;
  debug: boolean;
};

type SessionStoreEntry = {
  sessionId?: string;
  sessionFile?: string;
};

type CachedRoute = {
  channelId: string;
  accountId?: string;
  conversationId?: string;
  messageId?: string;
  replyToMessageId?: number;
  threadId?: number;
  expiresAt: number;
};

type PendingSuppression = {
  sessionKey: string;
  routeKey: string;
  expiresAt: number;
};

type PendingHandoff = {
  sessionKey: string;
  mediaPath: string;
  prependContext: string;
  expiresAt: number;
};

type ContextMessage = {
  role?: string;
  content?: unknown;
};

type GeminiContent = {
  role: "user" | "model";
  parts: Array<Record<string, unknown>>;
};

type DirectAudioResult = {
  heardText: string;
  intentText: string;
  tone: string;
  notes: string;
  confidence: string;
  replyText: string;
  rawText: string;
};

const DEFAULT_CONFIG: ResolvedPluginConfig = {
  provider: "haloai-gemini",
  model: "gemini-3-flash-preview",
  channels: ["telegram"],
  historyMessages: 12,
  maxHistoryChars: 12000,
  timeoutMs: 90000,
  onlyWhenNoText: true,
  mode: "handoff",
  requireStructuredResponse: true,
  suppressDefaultReply: true,
  suppressTranscriptEcho: true,
  storeHeardTextInSession: true,
  debug: false,
};

const DEFAULT_RUNTIME_STATE: PluginRuntimeState = {
  active: true,
  debug: false,
};

const ROUTE_TTL_MS = 10 * 60 * 1000;
const SUPPRESSION_TTL_MS = 90 * 1000;
const DEFAULT_FAILURE_NOTICE =
  "Direct audio handling failed for this voice message. Please try again or use text for this turn.";
const RUNTIME_STATE_FILENAME = "openclaw-voxsense.runtime.json";
const COMMAND_NAME = "voxsense";
const HEARD_CUSTOM_TYPES = ["contextual-gemini-audio.heard", "openclaw-voxsense.heard"] as const;

const routeCache = new Map<string, CachedRoute>();
const suppressionBySession = new Map<string, PendingSuppression>();
const handoffBySession = new Map<string, PendingHandoff>();
const cancelByRoute = new Map<string, number>();
const cancelBySession = new Map<string, number>();
const allowOwnSendByRoute = new Map<string, number>();
const activeRuns = new Set<string>();

function normalizeConfig(raw: unknown): ResolvedPluginConfig {
  const config = (raw ?? {}) as RawPluginConfig;
  return {
    provider: config.provider?.trim() || DEFAULT_CONFIG.provider,
    model: config.model?.trim() || DEFAULT_CONFIG.model,
    channels:
      Array.isArray(config.channels) && config.channels.length > 0
        ? config.channels.map((entry) => String(entry).trim().toLowerCase()).filter(Boolean)
        : DEFAULT_CONFIG.channels,
    historyMessages:
      typeof config.historyMessages === "number" && Number.isFinite(config.historyMessages)
        ? Math.max(0, Math.min(100, Math.floor(config.historyMessages)))
        : DEFAULT_CONFIG.historyMessages,
    maxHistoryChars:
      typeof config.maxHistoryChars === "number" && Number.isFinite(config.maxHistoryChars)
        ? Math.max(1000, Math.min(100000, Math.floor(config.maxHistoryChars)))
        : DEFAULT_CONFIG.maxHistoryChars,
    timeoutMs:
      typeof config.timeoutMs === "number" && Number.isFinite(config.timeoutMs)
        ? Math.max(1000, Math.min(600000, Math.floor(config.timeoutMs)))
        : DEFAULT_CONFIG.timeoutMs,
    onlyWhenNoText:
      typeof config.onlyWhenNoText === "boolean"
        ? config.onlyWhenNoText
        : DEFAULT_CONFIG.onlyWhenNoText,
    mode: config.mode === "reply" ? "reply" : DEFAULT_CONFIG.mode,
    requireStructuredResponse:
      typeof config.requireStructuredResponse === "boolean"
        ? config.requireStructuredResponse
        : DEFAULT_CONFIG.requireStructuredResponse,
    suppressDefaultReply:
      typeof config.suppressDefaultReply === "boolean"
        ? config.suppressDefaultReply
        : DEFAULT_CONFIG.suppressDefaultReply,
    suppressTranscriptEcho:
      typeof config.suppressTranscriptEcho === "boolean"
        ? config.suppressTranscriptEcho
        : DEFAULT_CONFIG.suppressTranscriptEcho,
    storeHeardTextInSession:
      typeof config.storeHeardTextInSession === "boolean"
        ? config.storeHeardTextInSession
        : DEFAULT_CONFIG.storeHeardTextInSession,
    debug: typeof config.debug === "boolean" ? config.debug : DEFAULT_CONFIG.debug,
    systemPrompt: config.systemPrompt?.trim() || undefined,
    failureNotice: config.failureNotice?.trim() || undefined,
  };
}

function stringifyMeta(meta: Record<string, unknown> | undefined): string {
  if (!meta || Object.keys(meta).length === 0) {
    return "";
  }
  try {
    return ` ${JSON.stringify(meta)}`;
  } catch {
    return "";
  }
}

function logInfo(api: OpenClawPluginApi, message: string, meta?: Record<string, unknown>): void {
  api.logger.info(`openclaw-voxsense: ${message}${stringifyMeta(meta)}`);
}

function logError(api: OpenClawPluginApi, message: string, meta?: Record<string, unknown>): void {
  api.logger.error(`openclaw-voxsense: ${message}${stringifyMeta(meta)}`);
}

function logDebug(
  api: OpenClawPluginApi,
  runtimeState: PluginRuntimeState,
  message: string,
  meta?: Record<string, unknown>,
): void {
  if (!runtimeState.debug) {
    return;
  }
  logInfo(api, message, meta);
}

function runtimeStatePath(api: OpenClawPluginApi): string {
  const stateDir = api.runtime.state.resolveStateDir(process.env);
  return path.join(stateDir, "plugins", RUNTIME_STATE_FILENAME);
}

async function readRuntimeState(
  api: OpenClawPluginApi,
  config: ResolvedPluginConfig,
): Promise<PluginRuntimeState> {
  const stored = await readJsonFile<Partial<PluginRuntimeState>>(runtimeStatePath(api));
  return {
    active: typeof stored?.active === "boolean" ? stored.active : DEFAULT_RUNTIME_STATE.active,
    debug: typeof stored?.debug === "boolean" ? stored.debug : config.debug,
  };
}

async function writeRuntimeState(
  api: OpenClawPluginApi,
  nextState: PluginRuntimeState,
): Promise<PluginRuntimeState> {
  const filePath = runtimeStatePath(api);
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, `${JSON.stringify(nextState, null, 2)}\n`, "utf8");
  return nextState;
}

function pruneState(): void {
  const now = Date.now();
  for (const [key, value] of routeCache) {
    if (value.expiresAt <= now) {
      routeCache.delete(key);
    }
  }
  for (const [key, value] of suppressionBySession) {
    if (value.expiresAt <= now) {
      suppressionBySession.delete(key);
    }
  }
  for (const [key, value] of handoffBySession) {
    if (value.expiresAt <= now) {
      handoffBySession.delete(key);
    }
  }
  for (const [key, expiresAt] of cancelByRoute) {
    if (expiresAt <= now) {
      cancelByRoute.delete(key);
    }
  }
  for (const [key, expiresAt] of cancelBySession) {
    if (expiresAt <= now) {
      cancelBySession.delete(key);
    }
  }
  for (const [key, expiresAt] of allowOwnSendByRoute) {
    if (expiresAt <= now) {
      allowOwnSendByRoute.delete(key);
    }
  }
}

function normalizeChannelId(value: unknown): string {
  return typeof value === "string" ? value.trim().toLowerCase() : "";
}

function asTrimmedString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function asFiniteNumber(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number.parseInt(value.trim(), 10);
    return Number.isFinite(parsed) ? parsed : undefined;
  }
  return undefined;
}

function routeCacheKey(channelId: string, messageId: string): string {
  return `${channelId}:${messageId}`;
}

function extractThreadId(value: unknown): number | undefined {
  if (!value || typeof value !== "object") {
    return undefined;
  }
  const record = value as Record<string, unknown>;
  return (
    asFiniteNumber(record.threadId) ??
    asFiniteNumber(record.messageThreadId) ??
    asFiniteNumber(record.topicId)
  );
}

function routeIdentityKey(params: {
  channelId: string;
  accountId?: string;
  conversationId?: string;
  threadId?: number;
}): string | null {
  const rawConversationId = params.conversationId?.trim();
  const channelId = params.channelId.trim().toLowerCase();
  const conversationId = rawConversationId?.replace(new RegExp(`^${channelId}:`, "i"), "");
  if (!conversationId) {
    return null;
  }
  return [
    channelId,
    params.accountId?.trim().toLowerCase() ?? "",
    conversationId,
    params.threadId != null ? String(params.threadId) : "",
  ].join(":");
}

function routeIdentityAliases(params: {
  channelId: string;
  accountId?: string;
  conversationId?: string;
  threadId?: number;
}): string[] {
  const primary = routeIdentityKey(params);
  if (!primary) {
    return [];
  }
  if (params.threadId == null) {
    return [primary];
  }
  const fallback = routeIdentityKey({ ...params, threadId: undefined });
  return fallback && fallback !== primary ? [primary, fallback] : [primary];
}

function routeIdentityKeyFromSending(
  event: PluginHookMessageSendingEvent,
  ctx: PluginHookMessageContext,
): string | null {
  return routeIdentityKey({
    channelId: ctx.channelId,
    accountId: ctx.accountId,
    conversationId: ctx.conversationId ?? event.to,
    threadId: extractThreadId(event.metadata),
  });
}

function parseAgentId(sessionKey: string): string {
  const match = /^agent:([^:]+):/i.exec(sessionKey.trim());
  return match?.[1]?.trim() || "main";
}

function sessionsStorePath(api: OpenClawPluginApi, agentId: string): string {
  const stateDir = api.runtime.state.resolveStateDir(process.env);
  return path.join(stateDir, "agents", agentId, "sessions", "sessions.json");
}

async function readJsonFile<T>(filePath: string): Promise<T | null> {
  try {
    const raw = await fs.readFile(filePath, "utf8");
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

async function loadSessionStoreEntry(
  api: OpenClawPluginApi,
  sessionKey: string,
): Promise<{ entry: SessionStoreEntry | null; sessionsDir: string }> {
  const agentId = parseAgentId(sessionKey);
  const storePath = sessionsStorePath(api, agentId);
  const store = await readJsonFile<Record<string, SessionStoreEntry>>(storePath);
  return {
    entry: store?.[sessionKey] ?? null,
    sessionsDir: path.dirname(storePath),
  };
}

function sessionFilePath(entry: SessionStoreEntry, sessionsDir: string): string | null {
  if (entry.sessionFile?.trim()) {
    return entry.sessionFile.trim();
  }
  if (!entry.sessionId?.trim()) {
    return null;
  }
  return path.join(sessionsDir, `${entry.sessionId.trim()}.jsonl`);
}

function textFromContent(content: unknown): string {
  if (typeof content === "string") {
    return content.trim();
  }
  if (!Array.isArray(content)) {
    return "";
  }
  const chunks: string[] = [];
  for (const part of content) {
    if (!part || typeof part !== "object") {
      continue;
    }
    const textValue = (part as { text?: unknown }).text;
    if (typeof textValue === "string" && textValue.trim()) {
      chunks.push(textValue.trim());
    }
  }
  return chunks.join("\n").trim();
}

function toGeminiRole(role: unknown): "user" | "model" | null {
  if (role === "assistant") {
    return "model";
  }
  if (role === "user") {
    return "user";
  }
  return null;
}

function collapseGeminiContents(contents: GeminiContent[]): GeminiContent[] {
  const merged: GeminiContent[] = [];
  for (const entry of contents) {
    const last = merged.at(-1);
    if (last && last.role === entry.role) {
      last.parts.push(...entry.parts);
      continue;
    }
    merged.push({ role: entry.role, parts: [...entry.parts] });
  }
  return merged;
}

function buildHistoryContents(messages: ContextMessage[], config: ResolvedPluginConfig): GeminiContent[] {
  const selected: GeminiContent[] = [];
  let totalChars = 0;
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    const role = toGeminiRole(message?.role);
    if (!role) {
      continue;
    }
    const text = textFromContent(message?.content);
    if (!text) {
      continue;
    }
    const nextChars = totalChars + text.length;
    if (selected.length >= config.historyMessages || nextChars > config.maxHistoryChars) {
      break;
    }
    totalChars = nextChars;
    selected.unshift({ role, parts: [{ text }] });
  }
  return collapseGeminiContents(selected);
}

type TranscriptEntry = {
  type?: string;
  id?: string;
  parentId?: string | null;
  customType?: string;
  data?: Record<string, unknown>;
  message?: {
    role?: string;
    content?: unknown;
  };
};

function parseTranscriptEntries(raw: string): TranscriptEntry[] {
  return raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .flatMap((line) => {
      try {
        const parsed = JSON.parse(line) as TranscriptEntry;
        return parsed && typeof parsed === "object" ? [parsed] : [];
      } catch {
        return [];
      }
    });
}

async function readTranscriptEntries(filePath: string): Promise<TranscriptEntry[]> {
  try {
    const raw = await fs.readFile(filePath, "utf8");
    return parseTranscriptEntries(raw);
  } catch {
    return [];
  }
}

function toContextMessage(entry: TranscriptEntry): ContextMessage | null {
  if (entry.type === "message" && entry.message && typeof entry.message === "object") {
    return {
      role: entry.message.role,
      content: entry.message.content,
    };
  }
  if (
    entry.type !== "custom" ||
    !HEARD_CUSTOM_TYPES.includes((entry.customType ?? "") as (typeof HEARD_CUSTOM_TYPES)[number])
  ) {
    return null;
  }
  const persistedAsUserMessage =
    entry.data && typeof entry.data === "object"
      ? (entry.data.persistedAsUserMessage as boolean | undefined) === true
      : false;
  if (persistedAsUserMessage) {
    return null;
  }
  const rawText =
    entry.data && typeof entry.data === "object"
      ? asTrimmedString(entry.data.text) ??
        asTrimmedString(entry.data.heardText) ??
        asTrimmedString(entry.data.transcript)
      : undefined;
  if (!rawText) {
    return null;
  }
  return {
    role: "user",
    content: [{ type: "text", text: rawText }],
  };
}

function lastTranscriptEntryId(entries: TranscriptEntry[]): string | null {
  for (let index = entries.length - 1; index >= 0; index -= 1) {
    const id = asTrimmedString(entries[index]?.id);
    if (id) {
      return id;
    }
  }
  return null;
}

function createTranscriptId(): string {
  return randomBytes(4).toString("hex");
}

async function loadHistoryContents(
  api: OpenClawPluginApi,
  sessionKey: string,
  config: ResolvedPluginConfig,
): Promise<GeminiContent[]> {
  const { entry, sessionsDir } = await loadSessionStoreEntry(api, sessionKey);
  const filePath = entry ? sessionFilePath(entry, sessionsDir) : null;
  if (!filePath) {
    return [];
  }
  const entries = await readTranscriptEntries(filePath);
  return buildHistoryContents(
    entries
      .map((entry) => toContextMessage(entry))
      .filter((entry): entry is ContextMessage => Boolean(entry)),
    config,
  );
}


function normalizeBaseUrl(raw: string | undefined): string {
  const trimmed = raw?.trim();
  if (!trimmed) {
    return "https://generativelanguage.googleapis.com/v1beta";
  }
  return trimmed.replace(/\/+$/, "");
}

function plainHeaders(input: unknown): Record<string, string> {
  if (!input || typeof input !== "object") {
    return {};
  }
  const result: Record<string, string> = {};
  for (const [key, value] of Object.entries(input)) {
    if (typeof value === "string" && value.trim()) {
      result[key] = value;
    }
  }
  return result;
}

async function resolveDirectAudioProvider(api: OpenClawPluginApi, config: ResolvedPluginConfig): Promise<{
  providerId: string;
  modelId: string;
  baseUrl: string;
  headers: Record<string, string>;
}> {
  const providerConfig = api.config.models?.providers?.[config.provider];
  if (!providerConfig) {
    throw new Error(`Unknown provider: ${config.provider}`);
  }
  if (providerConfig.api && providerConfig.api !== "google-generative-ai") {
    throw new Error(
      `Provider ${config.provider} uses api=${providerConfig.api}; direct audio mode currently needs google-generative-ai`,
    );
  }
  const auth = await api.runtime.modelAuth.resolveApiKeyForProvider({
    provider: config.provider,
    cfg: api.config,
  });
  const apiKey = requireApiKey(auth, config.provider);
  const headers = plainHeaders(providerConfig.headers);
  if (apiKey.startsWith("{")) {
    try {
      const parsed = JSON.parse(apiKey) as { token?: unknown };
      if (typeof parsed.token === "string" && parsed.token.trim()) {
        headers.Authorization = `Bearer ${parsed.token.trim()}`;
        headers["Content-Type"] = "application/json";
      }
    } catch {
      headers["x-goog-api-key"] = apiKey;
      headers["Content-Type"] = "application/json";
    }
  } else {
    headers["x-goog-api-key"] = apiKey;
    headers["Content-Type"] = "application/json";
  }
  return {
    providerId: config.provider,
    modelId: config.model,
    baseUrl: normalizeBaseUrl(providerConfig.baseUrl),
    headers,
  };
}

function buildDirectAudioPrompt(config: ResolvedPluginConfig, rawBody: string): string {
  const sections =
    config.mode === "reply"
      ? [
          "You are handling a new audio message inside an ongoing chat.",
          "Use the prior conversation history for context before interpreting the audio.",
          "Infer what the user said from the audio, then answer naturally as the assistant.",
          'Return only strict JSON with keys "heard_text" and "reply_text".',
          "reply_text must be the assistant's actual response, not a verbatim transcript of the audio, unless the user explicitly asked for transcription.",
          'If no audible speech is present, set "heard_text" to an empty string.',
          'If no assistant reply is needed, set "reply_text" to an empty string.',
          "Do not wrap the JSON in markdown fences.",
        ]
      : [
          "You are interpreting the user's latest voice message inside an ongoing chat.",
          "Use prior conversation history only to disambiguate the audio, never to replace it with a guess.",
          "Do not answer the user directly.",
          'Return only strict JSON with keys "heard_text", "intent_text", "tone", "notes", and "confidence".',
          "heard_text must be the best literal rendering of what the user said.",
          "intent_text must summarize what the user means in context while staying faithful to the audio.",
          "tone must briefly describe speaking style or emotion.",
          "notes must contain brief factual observations about emphasis, hesitation, laughter, language choice, or audio quality.",
          'confidence must be one of "high", "medium", or "low".',
          'If the audio is only a test or greeting, reflect that in intent_text instead of inventing a bigger request.',
          'If no audible speech is present, set both "heard_text" and "intent_text" to an empty string.',
          "Do not wrap the JSON in markdown fences.",
        ];
  if (rawBody && !isAudioPlaceholder(rawBody)) {
    sections.push(`The inbound message also had typed text or caption:\n${rawBody}`);
  }
  if (config.systemPrompt) {
    sections.push(config.systemPrompt);
  }
  return sections.join("\n\n");
}

function joinResponseText(payload: any): string {
  const parts = payload?.candidates?.[0]?.content?.parts;
  if (!Array.isArray(parts)) {
    return "";
  }
  return parts
    .map((part) => (typeof part?.text === "string" ? part.text.trim() : ""))
    .filter(Boolean)
    .join("\n")
    .trim();
}

function tryParseJsonObject(text: string): Record<string, unknown> | null {
  const trimmed = text.trim();
  if (!trimmed) {
    return null;
  }
  const candidates = [trimmed];
  const fenced = trimmed.match(/```(?:json)?\s*([\s\S]*?)```/i)?.[1]?.trim();
  if (fenced) {
    candidates.push(fenced);
  }
  const braces = trimmed.match(/\{[\s\S]*\}/)?.[0]?.trim();
  if (braces) {
    candidates.push(braces);
  }
  for (const candidate of candidates) {
    try {
      const parsed = JSON.parse(candidate) as Record<string, unknown>;
      if (parsed && typeof parsed === "object") {
        return parsed;
      }
    } catch {
      continue;
    }
  }
  return null;
}

function hasStructuredDirectAudioFields(
  parsed: Record<string, unknown>,
  mode: PluginMode,
): boolean {
  const hasHeardText =
    Object.prototype.hasOwnProperty.call(parsed, "heard_text") ||
    Object.prototype.hasOwnProperty.call(parsed, "heardText");
  const hasReplyText =
    Object.prototype.hasOwnProperty.call(parsed, "reply_text") ||
    Object.prototype.hasOwnProperty.call(parsed, "replyText");
  const hasIntentText =
    Object.prototype.hasOwnProperty.call(parsed, "intent_text") ||
    Object.prototype.hasOwnProperty.call(parsed, "intentText");
  return mode === "reply" ? hasHeardText && hasReplyText : hasHeardText && hasIntentText;
}

function parseDirectAudioResult(
  text: string,
  requireStructuredResponse: boolean,
  mode: PluginMode,
): DirectAudioResult | null {
  const parsed = tryParseJsonObject(text);
  if (!parsed) {
    if (requireStructuredResponse) {
      return null;
    }
    return {
      heardText: "",
      intentText: "",
      tone: "",
      notes: "",
      confidence: "",
      replyText: text.trim(),
      rawText: text,
    };
  }
  if (requireStructuredResponse && !hasStructuredDirectAudioFields(parsed, mode)) {
    return null;
  }
  const heardText = asTrimmedString(parsed.heard_text) ?? asTrimmedString(parsed.heardText) ?? "";
  const intentText =
    asTrimmedString(parsed.intent_text) ??
    asTrimmedString(parsed.intentText) ??
    heardText;
  const tone = asTrimmedString(parsed.tone) ?? "";
  const notes = asTrimmedString(parsed.notes) ?? "";
  const confidence = asTrimmedString(parsed.confidence) ?? "";
  const replyText = asTrimmedString(parsed.reply_text) ?? asTrimmedString(parsed.replyText) ?? "";
  return {
    heardText,
    intentText,
    tone,
    notes,
    confidence,
    replyText,
    rawText: text,
  };
}

function normalizeComparableText(value: string): string {
  return value
    .normalize("NFKC")
    .toLowerCase()
    .replace(/[\p{P}\p{S}]+/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function isTranscriptEcho(heardText: string, replyText: string): boolean {
  const heard = normalizeComparableText(heardText);
  const reply = normalizeComparableText(replyText);
  if (!heard || !reply) {
    return false;
  }
  if (heard === reply) {
    return true;
  }
  const shorter = heard.length <= reply.length ? heard : reply;
  const longer = heard.length <= reply.length ? reply : heard;
  return shorter.length >= 8 && longer.includes(shorter) && shorter.length / longer.length >= 0.85;
}

function truncateForLog(text: string, maxChars = 240): string {
  const trimmed = text.trim();
  if (trimmed.length <= maxChars) {
    return trimmed;
  }
  return `${trimmed.slice(0, maxChars)}…`;
}

async function callDirectAudioModel(params: {
  api: OpenClawPluginApi;
  config: ResolvedPluginConfig;
  history: GeminiContent[];
  mimeType: string;
  buffer: Buffer;
  rawBody: string;
}): Promise<{ result: DirectAudioResult; providerId: string; modelId: string }> {
  const provider = await resolveDirectAudioProvider(params.api, params.config);
  const url = `${provider.baseUrl}/models/${provider.modelId}:generateContent`;
  const contents: GeminiContent[] = [
    ...params.history,
    {
      role: "user",
      parts: [
        { text: buildDirectAudioPrompt(params.config, params.rawBody) },
        {
          inline_data: {
            mime_type: params.mimeType,
            data: params.buffer.toString("base64"),
          },
        },
      ],
    },
  ];

  const requestBodies = [
    {
      contents,
      generationConfig: {
        responseMimeType: "application/json",
      },
    },
    {
      contents,
    },
  ];

  let lastError = "unknown error";
  for (const body of requestBodies) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), params.config.timeoutMs);
    try {
      const response = await fetch(url, {
        method: "POST",
        headers: provider.headers,
        body: JSON.stringify(body),
        signal: controller.signal,
      });
      const rawText = await response.text();
      if (!response.ok) {
        lastError = `HTTP ${response.status}: ${rawText}`;
        continue;
      }
      const payload = JSON.parse(rawText);
      const responseText = joinResponseText(payload);
      if (!responseText) {
        lastError = "empty text response";
        continue;
      }
      const result = parseDirectAudioResult(
        responseText,
        params.config.requireStructuredResponse,
        params.config.mode,
      );
      if (!result) {
        lastError = `model did not return strict JSON: ${truncateForLog(responseText, 180)}`;
        continue;
      }
      return {
        result,
        providerId: provider.providerId,
        modelId: provider.modelId,
      };
    } catch (error) {
      lastError = error instanceof Error ? error.message : String(error);
    } finally {
      clearTimeout(timeout);
    }
  }
  throw new Error(`Direct audio request failed: ${lastError}`);
}

function isAudioPlaceholder(value: string): boolean {
  const trimmed = value.trim();
  if (!trimmed) {
    return true;
  }
  if (/^<media:audio>$/i.test(trimmed) || /^\[audio/i.test(trimmed) || /^🎤\s*\[audio/i.test(trimmed)) {
    return true;
  }
  const stripped = trimmed
    .replace(/^\[media attached:[^\n]*audio\/[^\n]*\]\s*/gi, "")
    .replace(
      /To send an image back, prefer the message tool[\s\S]*?Keep caption in the text body\.?/gi,
      "",
    )
    .replace(/Conversation info \(untrusted metadata\):\s*```[\s\S]*?```\s*/gi, "")
    .replace(/Sender \(untrusted metadata\):\s*```[\s\S]*?```\s*/gi, "")
    .replace(/<media:audio>/gi, "")
    // Strip additional metadata blocks that might be present
    .replace(/\[message_id:[^\n]*\]/gi, "")
    .replace(/\[topic:[^\n]*\]/gi, "")
    .replace(/\s+/g, " ")
    .trim();
  
  // If what remains is just punctuation or extremely short boilerplate, count it as a placeholder
  return stripped.length === 0 || /^[\s.,!?;:()[\]{}|\\/<>]+$/.test(stripped);
}

function looksLikeAudioTurn(params: {
  mediaType?: string;
  mediaPath?: string;
  rawBody: string;
}): boolean {
  if (params.mediaType?.toLowerCase().startsWith("audio/")) {
    return true;
  }
  if (params.mediaPath && /\.(aac|flac|m4a|mp3|oga|ogg|opus|wav|webm)$/i.test(params.mediaPath)) {
    return true;
  }
  return /<media:audio>/i.test(params.rawBody) || /^\[media attached:[^\n]*audio\//i.test(params.rawBody);
}

function shouldHandleAudioTurn(params: {
  config: ResolvedPluginConfig;
  channelId: string;
  mediaType?: string;
  mediaPath?: string;
  rawBody: string;
}): { ok: boolean; reason: string } {
  if (!params.config.channels.includes(params.channelId)) {
    return { ok: false, reason: "channel-not-enabled" };
  }
  if (!looksLikeAudioTurn(params)) {
    return { ok: false, reason: "not-audio" };
  }
  if (!params.config.onlyWhenNoText) {
    return { ok: true, reason: "matched" };
  }
  if (isAudioPlaceholder(params.rawBody)) {
    return { ok: true, reason: "matched" };
  }
  return { ok: false, reason: "typed-text-present" };
}

function splitTextForTelegram(text: string): string[] {
  const normalized = text.trim();
  if (!normalized) {
    return [];
  }
  const limit = 3500;
  const chunks: string[] = [];
  let remaining = normalized;
  while (remaining.length > limit) {
    const slice = remaining.slice(0, limit);
    const splitAt = Math.max(slice.lastIndexOf("\n\n"), slice.lastIndexOf("\n"), slice.lastIndexOf(" "));
    const cut = splitAt >= 1000 ? splitAt : limit;
    chunks.push(remaining.slice(0, cut).trim());
    remaining = remaining.slice(cut).trim();
  }
  if (remaining) {
    chunks.push(remaining);
  }
  return chunks;
}

function extractLatestUserPromptText(messages: unknown[]): string {
  if (!Array.isArray(messages)) {
    return "";
  }
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const entry = messages[index];
    if (!entry || typeof entry !== "object") {
      continue;
    }
    const role = (entry as { role?: unknown }).role;
    if (role !== "user") {
      continue;
    }
    const text = textFromContent((entry as { content?: unknown }).content);
    if (text) {
      return text;
    }
  }
  return "";
}

function extractAudioTurnFromPrompt(prompt: string, messages: unknown[]): {
  rawBody: string;
  mediaPath?: string;
  mediaType?: string;
} | null {
  const candidateTexts = [extractLatestUserPromptText(messages), prompt]
    .map((value) => value.trim())
    .filter(Boolean);
  for (const rawBody of candidateTexts) {
    const match =
      rawBody.match(/\[media attached:\s*([^|\]\n]+?)\s*\((audio\/[^)\n]+)\)\s*(?:\||\])/i) ??
      rawBody.match(/\[media attached:\s*([^|\]\n]+?)\s*(?:\||\])/i);
    const mediaPath = match?.[1]?.trim();
    const mediaType = match?.[2]?.trim();
    if (looksLikeAudioTurn({ mediaType, mediaPath, rawBody })) {
      return {
        rawBody,
        mediaPath,
        mediaType,
      };
    }
  }
  return null;
}

function buildAgentHandoffContext(result: DirectAudioResult): string {
  const lines = [
    "Audio understanding result for the user's latest voice message.",
    "Ignore raw placeholders like <media:audio> and attachment boilerplate for this turn.",
    `Heard text: ${result.heardText || "[unclear]"}`,
    `Intent in context: ${result.intentText || result.heardText || "[unclear]"}`,
  ];
  if (result.tone) {
    lines.push(`Tone: ${result.tone}`);
  }
  if (result.notes) {
    lines.push(`Notes: ${result.notes}`);
  }
  if (result.confidence) {
    lines.push(`Confidence: ${result.confidence}`);
  }
  lines.push(
    "Respond normally as the assistant for this chat. You may call tools and continue the normal multi-turn workflow.",
    "If the audio understanding looks uncertain, ask a brief clarification question instead of guessing.",
  );
  return lines.join("\n");
}

function buildAgentFailureContext(): string {
  return [
    "The user's latest turn is a voice message, but direct audio understanding failed for this turn.",
    "Do not pretend you understood the audio.",
    "Apologize briefly and ask the user to resend the voice message or type the request.",
  ].join("\n");
}

function buildStatusText(config: ResolvedPluginConfig, runtimeState: PluginRuntimeState): string {
  return [
    `Gemini audio direct mode: ${runtimeState.active ? "on" : "off"}`,
    `Mode: ${config.mode}`,
    `Debug logging: ${runtimeState.debug ? "on" : "off"}`,
    `Model: ${config.provider}/${config.model}`,
    `Suppress transcript echo: ${config.suppressTranscriptEcho ? "on" : "off"}`,
    `Store heard text in session: ${config.storeHeardTextInSession ? "on" : "off"}`,
    `Commands: /${COMMAND_NAME} status | /${COMMAND_NAME} on | /${COMMAND_NAME} off | /${COMMAND_NAME} debug on|off`,
  ].join("\n");
}

async function appendSessionArtifacts(params: {
  api: OpenClawPluginApi;
  sessionKey: string;
  heardText: string;
  intentText?: string;
  tone?: string;
  notes?: string;
  confidence?: string;
  replyText: string;
  providerId: string;
  modelId: string;
  storeHeardTextInSession: boolean;
  persistedAsUserMessage?: boolean;
}): Promise<void> {
  const { entry, sessionsDir } = await loadSessionStoreEntry(params.api, params.sessionKey);
  const filePath = entry ? sessionFilePath(entry, sessionsDir) : null;
  if (!filePath) {
    return;
  }
  try {
    await fs.access(filePath);
  } catch {
    return;
  }
  const existingEntries = await readTranscriptEntries(filePath);
  let parentId = lastTranscriptEntryId(existingEntries);
  const now = Date.now();
  const nextEntries: Array<TranscriptEntry & Record<string, unknown>> = [];
  if (params.storeHeardTextInSession && params.heardText.trim()) {
    const heardId = createTranscriptId();
    nextEntries.push({
      type: "custom",
      customType: "openclaw-voxsense.heard",
      data: {
        text: `[Audio heard directly by ${params.modelId}]\n${params.heardText.trim()}`,
        heardText: params.heardText.trim(),
        intentText: params.intentText?.trim() || undefined,
        tone: params.tone?.trim() || undefined,
        notes: params.notes?.trim() || undefined,
        confidence: params.confidence?.trim() || undefined,
        provider: params.providerId,
        model: params.modelId,
        persistedAsUserMessage: params.persistedAsUserMessage === true,
        createdAt: now,
      },
      id: heardId,
      parentId,
      timestamp: new Date(now).toISOString(),
    });
    parentId = heardId;
    const userTranscriptText = params.heardText.trim() || params.intentText?.trim() || "";
    if (params.persistedAsUserMessage === true && userTranscriptText) {
      const userId = createTranscriptId();
      nextEntries.push({
        type: "message",
        id: userId,
        parentId,
        timestamp: new Date(now).toISOString(),
        message: {
          role: "user",
          content: [{ type: "text", text: userTranscriptText }],
          timestamp: now,
          api: "google-generative-ai",
          provider: params.providerId,
          model: params.modelId,
        },
      });
      parentId = userId;
    }
  }
  if (params.replyText.trim()) {
    nextEntries.push({
      type: "message",
      id: createTranscriptId(),
      parentId,
      timestamp: new Date(now).toISOString(),
      message: {
        role: "assistant",
        content: [{ type: "text", text: params.replyText.trim() }],
        timestamp: now,
        api: "google-generative-ai",
        provider: params.providerId,
        model: params.modelId,
      },
    });
  }
  if (nextEntries.length === 0) {
    return;
  }
  const serialized = nextEntries.map((entry) => JSON.stringify(entry)).join("\n") + "\n";
  await fs.appendFile(filePath, serialized, "utf8");
}


async function sendTelegramReply(
  api: OpenClawPluginApi,
  route: CachedRoute,
  text: string,
): Promise<void> {
  const target = route.conversationId?.trim();
  if (!target) {
    throw new Error("Missing Telegram conversationId");
  }
  for (const chunk of splitTextForTelegram(text)) {
    const allowUntil = Date.now() + 10_000;
    for (const key of routeIdentityAliases(route)) {
      allowOwnSendByRoute.set(key, allowUntil);
    }
    await api.runtime.channel.telegram.sendMessageTelegram(target, chunk, {
      cfg: api.config,
      accountId: route.accountId,
      replyToMessageId: route.replyToMessageId,
      messageThreadId: route.threadId,
    });
  }
}

async function sendFailureNotice(
  api: OpenClawPluginApi,
  route: CachedRoute,
  config: ResolvedPluginConfig,
): Promise<void> {
  const notice = config.failureNotice?.trim() || DEFAULT_FAILURE_NOTICE;
  if (!notice) {
    return;
  }
  await sendTelegramReply(api, route, notice);
}

function cacheInboundRoute(event: PluginHookMessageReceivedEvent, ctx: PluginHookMessageContext): void {
  const messageId = asTrimmedString(event.metadata?.messageId);
  if (!messageId) {
    return;
  }
  const channelId = normalizeChannelId(ctx.channelId);
  if (!channelId) {
    return;
  }
  routeCache.set(routeCacheKey(channelId, messageId), {
    channelId,
    accountId: ctx.accountId,
    conversationId: ctx.conversationId,
    messageId,
    replyToMessageId: asFiniteNumber(messageId),
    threadId: extractThreadId(event.metadata),
    expiresAt: Date.now() + ROUTE_TTL_MS,
  });
}

function resolveCachedRoute(params: {
  channelId: string;
  messageId?: string;
  conversationId?: string;
}): CachedRoute | null {
  if (params.messageId) {
    const cached = routeCache.get(routeCacheKey(params.channelId, params.messageId));
    if (cached) {
      return cached;
    }
  }
  if (!params.conversationId?.trim()) {
    return null;
  }
  return {
    channelId: params.channelId,
    conversationId: params.conversationId.trim(),
    expiresAt: Date.now() + ROUTE_TTL_MS,
  };
}

export default function register(api: OpenClawPluginApi): void {
  const config = normalizeConfig(api.pluginConfig);

  const commandHandler = async (ctx: { args?: string }) => {
    const runtimeState = await readRuntimeState(api, config);
    const args = (ctx.args ?? "")
      .trim()
      .toLowerCase()
      .split(/\s+/)
      .filter(Boolean);
    if (args.length === 0 || args[0] === "status" || args[0] === "help") {
      return { text: buildStatusText(config, runtimeState) };
    }
    if (args[0] === "on" || args[0] === "off") {
      const nextState = await writeRuntimeState(api, {
        ...runtimeState,
        active: args[0] === "on",
      });
      logInfo(api, `runtime state updated via /${COMMAND_NAME}`, nextState);
      return { text: buildStatusText(config, nextState) };
    }
    if (args[0] === "debug" && (args[1] === "on" || args[1] === "off")) {
      const nextState = await writeRuntimeState(api, {
        ...runtimeState,
        debug: args[1] === "on",
      });
      logInfo(api, `runtime debug updated via /${COMMAND_NAME}`, nextState);
      return { text: buildStatusText(config, nextState) };
    }
    return { text: buildStatusText(config, runtimeState) };
  };

  api.registerCommand({
    name: COMMAND_NAME,
    description: "Manage VoxSense runtime mode for this plugin",
    acceptsArgs: true,
    handler: commandHandler,
  });

  api.on("message_received", async (event, ctx) => {
    pruneState();
    const runtimeState = await readRuntimeState(api, config);
    if (!runtimeState.active) {
      return;
    }
    cacheInboundRoute(event, ctx);
    logDebug(api, runtimeState, "cached inbound route", {
      channelId: ctx.channelId,
      conversationId: ctx.conversationId,
      messageId: asTrimmedString(event.metadata?.messageId),
    });
  });

  api.on("before_prompt_build", async (event, ctx): Promise<PluginHookBeforePromptBuildResult | void> => {
    pruneState();
    const runtimeState = await readRuntimeState(api, config);
    if (!runtimeState.active) {
      return;
    }
    const sessionKey = asTrimmedString(ctx.sessionKey);
    if (!sessionKey) {
      return;
    }
    if (config.mode === "reply") {
      const pending = suppressionBySession.get(sessionKey);
      if (!pending) {
        return;
      }
      suppressionBySession.delete(sessionKey);
      return {
        appendSystemContext: `A plugin is already handling this inbound audio turn out-of-band. Reply with exactly ${SILENT_REPLY_TOKEN}.`,
      };
    }
    const extracted = extractAudioTurnFromPrompt(event.prompt ?? "", event.messages ?? []);
    if (!extracted) {
      return;
    }
    const channelId = normalizeChannelId(ctx.channelId);
    const handleDecision = shouldHandleAudioTurn({
      config,
      channelId,
      mediaType: extracted.mediaType,
      mediaPath: extracted.mediaPath,
      rawBody: extracted.rawBody,
    });
    logDebug(api, runtimeState, "evaluated handoff audio turn", {
      sessionKey,
      channelId,
      mediaType: extracted.mediaType,
      mediaPath: extracted.mediaPath,
      decision: handleDecision.reason,
      promptPreview: truncateForLog(extracted.rawBody, 180),
    });
    if (!handleDecision.ok) {
      return;
    }
    if (!extracted.mediaPath) {
      return {
        prependContext: buildAgentFailureContext(),
      };
    }
    const cached = handoffBySession.get(sessionKey);
    if (cached && cached.mediaPath === extracted.mediaPath && cached.expiresAt > Date.now()) {
      return { prependContext: cached.prependContext };
    }
    try {
      const [history, buffer] = await Promise.all([
        loadHistoryContents(api, sessionKey, config),
        fs.readFile(extracted.mediaPath),
      ]);
      const { result, providerId, modelId } = await callDirectAudioModel({
        api,
        config,
        history,
        mimeType: extracted.mediaType || "audio/ogg",
        buffer,
        rawBody: extracted.rawBody,
      });
      const prependContext = buildAgentHandoffContext(result);
      handoffBySession.set(sessionKey, {
        sessionKey,
        mediaPath: extracted.mediaPath,
        prependContext,
        expiresAt: Date.now() + SUPPRESSION_TTL_MS,
      });
      await appendSessionArtifacts({
        api,
        sessionKey,
        heardText: result.heardText,
        intentText: result.intentText,
        tone: result.tone,
        notes: result.notes,
        confidence: result.confidence,
        replyText: "",
        providerId,
        modelId,
        storeHeardTextInSession: config.storeHeardTextInSession,
        persistedAsUserMessage: config.storeHeardTextInSession,
      });
      logInfo(api, `prepared handoff audio turn for ${sessionKey}`, {
        heardPreview: truncateForLog(result.heardText, 160),
        intentPreview: truncateForLog(result.intentText, 160),
        tone: result.tone || undefined,
        confidence: result.confidence || undefined,
      });
      return { prependContext };
    } catch (error) {
      logError(api, `handoff audio understanding failed for ${sessionKey}`, {
        error: error instanceof Error ? error.message : String(error),
        mediaPath: extracted.mediaPath,
        mediaType: extracted.mediaType,
      });
      return {
        prependContext: buildAgentFailureContext(),
      };
    }
  });

  api.on(
    "message_sending",
    async (event, ctx): Promise<PluginHookMessageSendingResult | void> => {
      pruneState();
      const runtimeState = await readRuntimeState(api, config);
      if (!runtimeState.active) {
        return;
      }
      if (config.mode !== "reply") {
        return;
      }
      const routeKey = routeIdentityKeyFromSending(event, ctx);
      if (!routeKey) {
        return;
      }
      const allowOwnSendUntil = allowOwnSendByRoute.get(routeKey);
      if (allowOwnSendUntil && allowOwnSendUntil > Date.now()) {
        allowOwnSendByRoute.delete(routeKey);
        logDebug(api, runtimeState, "allowed plugin-owned outbound send", {
          routeKey,
          preview: truncateForLog(event.content, 120),
        });
        return;
      }
      const expiresAt = cancelByRoute.get(routeKey);
      if (!expiresAt || expiresAt <= Date.now()) {
        return;
      }
      logInfo(api, `cancelled default outbound send for ${routeKey}`);
      return { cancel: true };
    },
  );

  api.on("before_tool_call", async (event, ctx) => {
    pruneState();
    const runtimeState = await readRuntimeState(api, config);
    if (!runtimeState.active) {
      return;
    }
    if (config.mode !== "reply") {
      return;
    }
    const sessionKey = asTrimmedString(ctx.sessionKey);
    if (!sessionKey) {
      return;
    }
    const cancelUntil = cancelBySession.get(sessionKey);
    if (!cancelUntil || cancelUntil <= Date.now()) {
      return;
    }
    logInfo(api, `blocked default tool call during direct audio turn for ${sessionKey}`, {
      toolName: event.toolName,
    });
    return {
      block: true,
      blockReason: "This audio turn is already being handled by the direct-audio plugin.",
    };
  });

  api.registerHook(
    "message:preprocessed",
    async (event: any) => {
      pruneState();
      const runtimeState = await readRuntimeState(api, config);
      if (!runtimeState.active) {
        return;
      }
      if (config.mode !== "reply") {
        return;
      }
      if (!event || event.type !== "message" || event.action !== "preprocessed") {
        return;
      }
      const sessionKey = asTrimmedString(event.sessionKey);
      if (!sessionKey || activeRuns.has(sessionKey)) {
        return;
      }
      const context = (event.context ?? {}) as Record<string, unknown>;
      const channelId = normalizeChannelId(context.channelId);
      const body = asTrimmedString(context.body) ?? "";
      const bodyForAgent = asTrimmedString(context.bodyForAgent) ?? "";
      const rawBody = bodyForAgent || body;
      const mediaType = asTrimmedString(context.mediaType);
      const mediaPath = asTrimmedString(context.mediaPath);
      const handleDecision = shouldHandleAudioTurn({
        config,
        channelId,
        mediaType,
        mediaPath,
        rawBody,
      });
      if (looksLikeAudioTurn({ mediaType, mediaPath, rawBody })) {
        logDebug(api, runtimeState, "evaluated preprocessed audio turn", {
          sessionKey,
          channelId,
          mediaType,
          mediaPath,
          bodyPreview: truncateForLog(body, 160),
          bodyForAgentPreview: truncateForLog(bodyForAgent, 160),
          decision: handleDecision.reason,
        });
      }
      if (!handleDecision.ok) {
        return;
      }
      if (!mediaPath) {
        logDebug(api, runtimeState, "skipped direct audio turn without mediaPath", {
          sessionKey,
          channelId,
          mediaType,
        });
        return;
      }
      const route = resolveCachedRoute({
        channelId,
        messageId: asTrimmedString(context.messageId),
        conversationId: asTrimmedString(context.conversationId),
      });
      if (!route || route.channelId !== "telegram") {
        logDebug(api, runtimeState, "skipped direct audio turn without telegram route", {
          sessionKey,
          channelId,
          conversationId: asTrimmedString(context.conversationId),
          messageId: asTrimmedString(context.messageId),
        });
        return;
      }
      const routeKey = routeIdentityKey(route);
      if (!routeKey) {
        return;
      }
      activeRuns.add(sessionKey);
      if (config.suppressDefaultReply) {
        suppressionBySession.set(sessionKey, {
          sessionKey,
          routeKey,
          expiresAt: Date.now() + SUPPRESSION_TTL_MS,
        });
        const cancelUntil = Date.now() + SUPPRESSION_TTL_MS;
        cancelBySession.set(sessionKey, cancelUntil);
        for (const key of routeIdentityAliases(route)) {
          cancelByRoute.set(key, cancelUntil);
        }
      }
      logInfo(api, `handling direct audio turn for ${sessionKey}`, {
        channelId,
        mediaType,
        conversationId: route.conversationId,
        threadId: route.threadId,
      });
      try {
        const [history, buffer] = await Promise.all([
          loadHistoryContents(api, sessionKey, config),
          fs.readFile(mediaPath),
        ]);
        logDebug(api, runtimeState, "loaded direct audio inputs", {
          sessionKey,
          historyTurns: history.length,
          mediaBytes: buffer.byteLength,
          mediaType,
          mediaPath,
        });
        const { result, providerId, modelId } = await callDirectAudioModel({
          api,
          config,
          history,
          mimeType: mediaType || "audio/ogg",
          buffer,
          rawBody,
        });
        logDebug(api, runtimeState, "model returned direct audio result", {
          sessionKey,
          heardPreview: truncateForLog(result.heardText),
          intentPreview: truncateForLog(result.intentText),
          replyPreview: truncateForLog(result.replyText),
          rawPreview: truncateForLog(result.rawText),
        });
        let chatReplyText = result.replyText.trim();
        if (
          config.suppressTranscriptEcho &&
          chatReplyText &&
          isTranscriptEcho(result.heardText, chatReplyText)
        ) {
          logInfo(api, `suppressed transcript-like chat reply for ${sessionKey}`, {
            heardPreview: truncateForLog(result.heardText, 120),
            replyPreview: truncateForLog(chatReplyText, 120),
          });
          chatReplyText = "";
        }
        if (chatReplyText) {
          await sendTelegramReply(api, route, chatReplyText);
        }
        await appendSessionArtifacts({
          api,
          sessionKey,
          heardText: result.heardText,
          intentText: result.intentText,
          tone: result.tone,
          notes: result.notes,
          confidence: result.confidence,
          replyText: chatReplyText,
          providerId,
          modelId,
          storeHeardTextInSession: config.storeHeardTextInSession,
        });
        logInfo(api, `direct audio turn completed for ${sessionKey}`, {
          providerId,
          modelId,
          heardText: Boolean(result.heardText.trim()),
          replied: Boolean(chatReplyText),
        });
      } catch (error) {
        logError(api, `direct audio turn failed for ${sessionKey}`, {
          error: error instanceof Error ? error.message : String(error),
          channelId,
          mediaType,
          mediaPath,
        });
        if (config.suppressDefaultReply) {
          try {
            await sendFailureNotice(api, route, config);
          } catch (sendError) {
            logError(api, "failed to send failure notice", {
              error: sendError instanceof Error ? sendError.message : String(sendError),
            });
          }
        }
      } finally {
        activeRuns.delete(sessionKey);
      }
    },
    {
      name: "openclaw-voxsense-preprocessed",
      description: "Handle inbound audio turns with direct contextual multimodal inference",
    },
  );
}

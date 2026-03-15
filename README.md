# OpenClaw VoxSense

Context-aware voice understanding for OpenClaw.  
面向 OpenClaw 的上下文语音理解插件。

Current channel support: Telegram only.  
当前仅支持 Telegram。

Current model support: Gemini only for now.  
当前模型支持暂时仅限 Gemini。

## Demo / 演示

**Prompt / 问题**

`我是个大帅哥能告诉我这句话里哪个字最长吗`

**ASR only (`Groq / whisper-large-v3-turbo`)**

![ASR-only demo](./assets/asr-answer.gif)

**VoxSense (`Gemini-3-Flash`)**

![VoxSense demo](./assets/voxsense.gif)

## Summary / 简介

**English**

- Works with `OpenClaw >= 2026.3.11`
- Currently supports Gemini multimodal models only
- Default mode is `handoff`
- A multimodal model listens to the raw voice message plus recent chat context
- The plugin returns structured understanding to the main OpenClaw agent
- Normal tool calling, memory, and TTS reply flow stay available

**中文**

- 适用于 `OpenClaw >= 2026.3.11`
- 当前暂时仅支持 Gemini 多模态模型
- 默认模式为 `handoff`
- 使用多模态模型直接理解原始语音和最近会话上下文
- 插件把结构化理解结果交回主 OpenClaw agent
- 正常工具调用、记忆、多轮和 TTS 回复链路仍然可用

## Comparison / 对比

| Item | OpenClaw built-in voice (current) | VoxSense |
| --- | --- | --- |
| Input path | `tools.media.audio` transcribes voice to text first | Multimodal model reads raw audio directly |
| What the agent receives | Transcript text | Transcript + intent + tone + notes + confidence |
| Context use during understanding | Mostly after transcription | During audio understanding itself |
| Tool calling | Available | Available in `handoff` mode |
| Multi-turn chat | Available | Available in `handoff` mode |
| Voice reply | Uses normal OpenClaw TTS | Uses normal OpenClaw TTS |
| Best for | Simple STT-first voice chat | Context-aware voice understanding |
| Tradeoff | Simpler and more predictable | Richer understanding, but more model-dependent |

## Install / 安装

### From npm / 从 npm 安装

```bash
openclaw plugins install openclaw-voxsense
```

### From source / 从源码加载

```json
{
  "hooks": {
    "internal": {
      "enabled": true
    }
  },
  "plugins": {
    "load": {
      "paths": [
        "~/.openclaw/workspace/plugins/openclaw-voxsense"
      ]
    },
    "entries": {
      "openclaw-voxsense": {
        "enabled": true,
        "hooks": {
          "allowPromptInjection": true
        },
        "config": {
          "provider": "haloai-gemini",
          "model": "gemini-3-flash-preview",
          "mode": "handoff",
          "onlyWhenNoText": true,
          "storeHeardTextInSession": true,
          "debug": false
        }
      }
    }
  }
}
```

Restart the gateway after config changes.  
修改配置后需要重启网关。

## TTS / 语音回复

VoxSense does not synthesize speech by itself.  
VoxSense 本身不负责语音合成。

For voice replies, use normal OpenClaw TTS:

- configure `messages.tts`
- or enable it per session with `/tts always`

如果你希望机器人“说话”，仍然需要使用 OpenClaw 自带的 TTS：

- 配置 `messages.tts`
- 或者在会话里用 `/tts always`

## Key Config / 关键配置

- `provider`: provider id used for direct audio understanding
- `model`: model id used for direct audio understanding
- `mode`:
  - `handoff`: recommended; understand voice, then hand the turn back to the main agent
  - `reply`: legacy direct-reply mode
- `onlyWhenNoText`: only intercept pure voice turns
- `storeHeardTextInSession`: persist understood voice content into session history
- `debug`: verbose plugin logs

## Runtime Command / 运行时命令

Primary command:

```text
/voxsense status
/voxsense on
/voxsense off
/voxsense debug on
/voxsense debug off
```

## Notes / 说明

- VoxSense currently supports Gemini-family models exposed through the `google-generative-ai` `generateContent` shape
- Other multimodal providers are not supported yet
- If you only want VoxSense, you can disable the built-in STT path with `tools.media.audio.enabled=false`
- If built-in STT stays enabled, both paths may run and cost more

## Publish / 发布

Recommended release path:

1. push source to GitHub
2. publish package to npm
3. let users install with `openclaw plugins install openclaw-voxsense`

推荐发布方式：

1. 源码放 GitHub
2. npm 发布包
3. 用户通过 `openclaw plugins install openclaw-voxsense` 安装

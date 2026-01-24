# MorningGoal

MorningGoal 是一个帮助用户建立早晨习惯的 iOS 应用。

## 🛠 开发环境配置 (Development Setup)

本项目使用 **Fastlane Match** 来管理 iOS 证书和 Provisioning Profile，确保团队成员拥有一致的签名环境。

### 1. 前置要求

确保你已经安装了以下工具：
- Ruby (建议使用 rbenv 或 rvm)
- Bundler (`gem install bundler`)
- Xcode

### 2. 初始化依赖

在项目根目录下运行：

```bash
bundle install
```

### 3. 配置 App Store Connect API Key

为了能够下载证书和上传应用，你需要配置 App Store Connect API Key。

1. 复制环境变量模版：
   ```bash
   cp fastlane/.env.sample .env
   ```
   *(注意：`.env` 文件已被 git 忽略，请勿提交)*

2. 编辑 `.env` 文件，填入你的 API Key 信息：
   - `APP_STORE_CONNECT_API_KEY_KEY_ID`
   - `APP_STORE_CONNECT_API_KEY_ISSUER_ID`
   - `APP_STORE_CONNECT_API_KEY_KEY_CONTENT` (Base64 编码的 .p8 文件内容)

### 4. 同步证书与签名 (Provisioning Profiles)

本项目使用私有 Git 仓库存储加密的证书。你需要获得仓库的访问权限以及解密密码（Passphrase）。

运行以下命令一键配置 Xcode 签名设置：

```bash
bundle exec fastlane setup_signing
```

该命令会自动：
1. 从 Git 仓库下载最新的证书和 Profile。
2. 安装到你的 Keychain 和 Xcode 中。
3. 自动修改 Xcode 项目设置，关闭“自动签名”，并绑定到正确的 Profile。

### 5. 常用命令

| 命令 | 说明 |
| --- | --- |
| `bundle exec fastlane setup_signing` | **推荐**：同步证书并自动配置 Xcode 签名设置 |
| `bundle exec fastlane certificates` | 仅同步证书，不修改 Xcode 设置 |
| `bundle exec fastlane beta` | 打包并上传 TestFlight (自动递增 Build 号) |
| `bundle exec fastlane daily_build` | 每日构建流程 (测试 + 打包 + TestFlight) |
| `bundle exec fastlane release` | 打包并上传 App Store (仅二进制) |

## ⚠️ 常见问题

### 证书数量达到上限 (Max number of certificates)

如果你遇到 `reached the maximum number of available Distribution certificates` 错误，说明 Apple 账号上的证书数量已满（通常个人账号限制 3 个），且本地/仓库中没有可用的证书。

**解决方案**：
运行以下命令重置证书（**警告**：这会撤销该账号下所有 App 的现有证书，请谨慎操作）：

```bash
bundle exec fastlane reset_certificates
```

---

## 📚 文档

- [Match 多项目复用指南](docs/Match_Integration_Guide.md): 如何在其他项目中使用同一个证书仓库。

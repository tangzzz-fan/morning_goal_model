# Fastlane Match 多项目复用指南

本文档介绍如何在其他 iOS 项目中复用现有的证书仓库 (`tangorios_apple_cert`)，实现统一的证书管理。

## 什么是 Match 复用？

Fastlane Match 允许将证书和 Provisioning Profile 存储在一个私有的 Git 仓库中。默认情况下，一个仓库可以管理多个不同 Bundle ID 的应用的证书。

只要这些应用属于同一个 Apple Developer Team，就可以（也建议）共享同一个 Match 仓库。这样做的好处是：
- 只需要维护一套 Git 仓库访问权限。
- 只需要记住一个解密密码（Passphrase）。
- 证书（Certificate）可以被多个 App 共享（虽然 Profile 是每个 App 独立的）。

## 集成步骤

### 1. 初始化 Fastlane (如果尚未初始化)

在项目根目录下运行：

```bash
fastlane init
```

### 2. 创建 Matchfile

在 `fastlane/Matchfile` 中指定同一个 Git 仓库地址。

**示例内容**：

```ruby
git_url("git@github.com:tangzzz-fan/tangorios_apple_cert.git")

storage_mode("git")

type("development") # 默认类型

# 你的新 App 的 Bundle ID
app_identifier(["com.yourcompany.newapp"]) 

# 这里的用户名通常是创建证书的 Apple ID，但在使用 API Key 时不重要
username("user@example.com") 
```

### 3. 配置 Fastfile

在 `fastlane/Fastfile` 中添加 lane 来同步证书。

**基础配置**：

```ruby
platform :ios do
  desc "Sync certificates"
  lane :certificates do
    # readonly: true 表示仅下载，不创建新证书/Profile
    # 如果是 CI/CD 环境，建议开启 readonly: true
    # 如果是本地开发且需要创建新 Profile，设为 false
    match(type: "development", readonly: false)
    match(type: "appstore", readonly: false)
  end
end
```

### 4. 运行同步

```bash
bundle exec fastlane certificates
```

### 5. 首次运行注意事项

1. **Passphrase**: 首次运行时，终端会询问解密密码。**必须输入与 MorningGoal 项目相同的密码**，否则无法解密仓库内容。
2. **App ID 创建**: 如果你的新 App Bundle ID 在 Apple Developer Portal 上还不存在，`match` 会自动尝试创建它（如果权限足够）。
3. **Profile 创建**: `match` 会检测到仓库里没有这个新 Bundle ID 的 Profile，于是会自动生成一个新的，并上传到 Git 仓库。

## 最佳实践

### 关于证书 (Certificates)
Apple 限制每个账号的发布证书数量（通常 3 个）。
- **Match 的优势**：当多个项目共用一个 Match 仓库时，Match 会智能地**复用**现有的有效证书，而不是为每个项目创建新证书。
- 这避免了“证书数量达到上限”的问题。

### 团队协作
- 确保所有团队成员都有 Git 仓库的读写权限。
- 将 Passphrase 安全地共享给团队成员（例如通过 1Password）。

### CI/CD 集成
在 CI 环境中，通常只需要**读取**权限：

```ruby
match(type: "appstore", readonly: true)
```

这样可以防止 CI 意外撤销或修改现有证书。

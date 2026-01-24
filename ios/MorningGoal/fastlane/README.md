fastlane documentation
----

# Installation

Make sure you have the latest version of the Xcode command line tools installed:

```sh
xcode-select --install
```

For _fastlane_ installation instructions, see [Installing _fastlane_](https://docs.fastlane.tools/#installing-fastlane)

# Available Actions

## iOS

### ios certificates

```sh
[bundle exec] fastlane ios certificates
```

Certificates

### ios setup_signing

```sh
[bundle exec] fastlane ios setup_signing
```

Update Xcode Signing settings to use Match profiles

### ios build

```sh
[bundle exec] fastlane ios build
```

Build the app

### ios beta

```sh
[bundle exec] fastlane ios beta
```

Push a new beta build to TestFlight

### ios release

```sh
[bundle exec] fastlane ios release
```

Release to App Store Connect

### ios daily_build

```sh
[bundle exec] fastlane ios daily_build
```

Daily Build (Runs tests, increments build number, uploads to TestFlight)

### ios reset_certificates

```sh
[bundle exec] fastlane ios reset_certificates
```

Reset Certificates (Use when 'Max number of certificates' error occurs)

----

This README.md is auto-generated and will be re-generated every time [_fastlane_](https://fastlane.tools) is run.

More information about _fastlane_ can be found on [fastlane.tools](https://fastlane.tools).

The documentation of _fastlane_ can be found on [docs.fastlane.tools](https://docs.fastlane.tools).

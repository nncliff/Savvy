# Mobile App Development Guide

## Overview
This mobile app has two development modes due to custom native plugins that are incompatible with Expo Go.

## Configuration Files
- **`app.json`** - Full configuration with custom plugins (requires development build)
- **`app.dev.json`** - Expo Go compatible configuration (for quick testing)
- **`babel.config.js`** - Configured with react-native-reanimated plugin

## Setup After Cloning Repository

### 1. Install Dependencies
```bash
cd /workspaces/Savvy
pnpm install
```

### 2. Navigate to Mobile App
```bash
cd apps/mobile
```

## Development Workflows

### Option A: Quick Testing with Expo Go (Recommended for daily development)

**Use this for:** Fast iteration, UI testing, basic functionality testing

1. **Start development server:**
   ```bash
   npx expo start --config app.dev.json --tunnel
   ```

2. **Connect your device:**
   - Install Expo Go app on your Android device
   - Scan the QR code displayed in terminal
   - Or press `s` to switch to Expo Go mode if needed

**Features Available:**
- ✅ Basic app functionality
- ✅ Navigation and UI
- ✅ Most core features
- ❌ Share intent functionality
- ❌ Custom native plugins

### Option B: Full Feature Testing with Development Build

**Use this for:** Testing custom plugins, share functionality, final testing before release

1. **Install EAS CLI (one-time setup):**
   ```bash
   npm install -g @expo/eas-cli
   eas login
   ```

2. **Build development build:**
   ```bash
   npx expo install --fix
   eas build --profile development --platform android
   ```

3. **Install the generated APK on your device**

4. **Start development server:**
   ```bash
   npx expo start --config app.json --tunnel
   ```

5. **Connect via development build app (not Expo Go)**

**Features Available:**
- ✅ All app functionality
- ✅ Share intent from other apps
- ✅ Custom native plugins
- ✅ All native configurations

## Troubleshooting

### Common Issues

#### "Reanimated Native part not initialized"
- **Solution:** Make sure `babel.config.js` has the reanimated plugin as the last plugin
- Clear cache: `npx expo start --clear`

#### "Missing default export" warnings
- **Solution:** These are usually false positives after cache clearing, they should resolve automatically

#### App won't load in Expo Go
- **Cause:** You're trying to use `app.json` (which has custom plugins) with Expo Go
- **Solution:** Use `app.dev.json` configuration instead

#### Tunnel connection issues
- **Solution:** Make sure you're using `--tunnel` flag for remote device testing

### Cache Clearing (if needed)
```bash
# Clear Expo cache
rm -rf .expo

# Clear Metro cache and restart
npx expo start --clear
```

## Recommended Development Flow

1. **Daily Development:** Use Option A (Expo Go + `app.dev.json`)
2. **Feature Testing:** Use Option A for most features
3. **Integration Testing:** Use Option B (Development Build + `app.json`) before major releases
4. **Share Feature Testing:** Use Option B when testing share intent functionality

## Package.json Scripts (Optional)

Add these to your `package.json` for convenience:

```json
{
  "scripts": {
    "start:expo-go": "expo start --config app.dev.json --tunnel",
    "start:dev-build": "expo start --config app.json --tunnel",
    "build:android": "eas build --profile development --platform android"
  }
}
```

## Key Points to Remember

1. **Two configs exist for a reason** - don't delete either one
2. **Expo Go** = fast iteration but limited features
3. **Development Build** = full features but slower build process
4. **Always use `--tunnel`** for testing on physical devices
5. **Babel config** must keep reanimated plugin as the last plugin

## Version Compatibility

- Expo SDK: ~52.0.11
- React Native: 0.76.3
- React Native Reanimated: ^3.16.2

These versions are tested and working together.

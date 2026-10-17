# Modrinth listing – copy & paste

Everything you need to fill in on modrinth.com for **PyMC Bot**. Files referenced here are in this folder.

## 1. Create the project (modrinth.com → **+** → *Create a project*)

| Field | Value |
| --- | --- |
| Project type | **Mod** |
| Name | `PyMC Bot` |
| URL (slug) | `pymcbot` *(`pymc-bot` is already taken)* |
| Visibility | Public |
| Summary | `Record your own Minecraft playtime and train a bot that plays like you. Client-side, private, one jar per version from 1.21.1 to 26.3.` |

## 2. Icon

Upload **`icon.png`** (512×512, 146 KB – the limit is 256 KiB).

## 3. Description

Paste the whole of [`../../mod/MODRINTH.md`](../../mod/MODRINTH.md) into *Description*
(or run `python scripts/modrinth_publish.py --project pymcbot --update-page` with a token).
The banner at the top loads from GitHub once PR #1 is merged into `main`. To show it before that, upload
`banner.jpg` to the gallery first, copy its image link and replace the first line of the description with it.

## 4. Settings → General / Tags

| Field | Value |
| --- | --- |
| Environment | **Client-side only** (client: required, server: unsupported) |
| Primary categories | **Utility** |
| Additional categories | **Technology** |
| License | **MIT** (`MIT`) |

## 5. Settings → Links

| Field | Value |
| --- | --- |
| Source code | `https://github.com/codero-sus/PyMC_Bot` |
| Issue tracker | `https://github.com/codero-sus/PyMC_Bot/issues` |
| Wiki | `https://github.com/codero-sus/PyMC_Bot/blob/main/mod/README.md` |
| Discord | *(leave empty)* |

## 6. Gallery

| Image | Title | Description | Featured |
| --- | --- | --- | --- |
| `banner.jpg` | `PyMC Bot` | `Record how you play, then let a bot play like you.` | ✅ |

## 7. Versions (one per Minecraft version)

Download the 16 jars from the GitHub release
[`pymcbot-v0.7.0`](https://github.com/codero-sus/PyMC_Bot/releases/tag/pymcbot-v0.7.0), then either:

- **Automatic:** `MODRINTH_TOKEN=mrp_... python scripts/modrinth_publish.py --project pymcbot --jars <folder>`
  (token: modrinth.com/settings/pats, scopes *Create versions* + *Write projects*), or
- **By hand**, for each jar:

| Field | Value |
| --- | --- |
| File | `pymcbot-0.7.0+<mc>.jar` |
| Version number | `0.7.0+<mc>` (e.g. `0.7.0+26.3`) |
| Version title | `PyMC Bot 0.7.0 for Minecraft <mc>` |
| Channel | Release |
| Loaders | Fabric |
| Game versions | only the `<mc>` in the file name |
| Dependencies | Fabric API – **required** · Mod Menu – **optional** |
| Changelog | the `## 0.7.0` section of [`../../mod/CHANGELOG.md`](../../mod/CHANGELOG.md) |

## 8. Submit

*Submit for review*. Moderators usually check that the description explains what the mod does, the
license and source link are set and the files match the description. Everything above covers that.

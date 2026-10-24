# Modrinth listing – copy & paste

Everything you need to fill in on modrinth.com for **PyMC Bot**. All pictures load from the root of the `main` branch
(`icon.png`, `banner.jpg` – the same files as in this folder).

![PyMC Bot logo](https://raw.githubusercontent.com/codero-sus/PyMC_Bot/main/icon.png)

![PyMC Bot banner](https://raw.githubusercontent.com/codero-sus/PyMC_Bot/main/banner.jpg)

## 1. Create the project (modrinth.com → **+** → *Create a project*)

| Field | Value |
| --- | --- |
| Project type | **Mod** |
| Name | `PyMC Bot` |
| URL (slug) | `pymcbot` *(`pymc-bot` is already taken)* |
| Visibility | Public |
| Summary | `Train a bot that plays like you, then control your PyMC Bot bots in-game: 44 skills, auto-sleep, autopilot. Client-side, private, one jar per version from 1.21.1 to 26.3.` |

## 2. Icon

Upload the logo (512×512, 146 KB – the limit is 256 KiB): download it from
https://raw.githubusercontent.com/codero-sus/PyMC_Bot/main/icon.png

## 3. Description

Paste the whole of [`../../mod/MODRINTH.md`](../../mod/MODRINTH.md) into *Description*
(after the GitHub connection below, every release from `main` re-syncs it automatically).
The banner at the top of the description is https://raw.githubusercontent.com/codero-sus/PyMC_Bot/main/banner.jpg –
it is in the root of the `main` branch, so it shows on Modrinth right away.

## 4. Settings → General / Tags

| Field | Value |
| --- | --- |
| Environment | **Client-side only** (client: required, server: unsupported) |
| Primary categories | **Utility** |
| Additional categories | **Technology** |
| License | **PolyForm Noncommercial 1.0.0** – pick *Custom license* / type the SPDX id `PolyForm-Noncommercial-1.0.0`, license URL `https://polyformproject.org/licenses/noncommercial/1.0.0` |

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
| [banner.jpg](https://raw.githubusercontent.com/codero-sus/PyMC_Bot/main/banner.jpg) | `PyMC Bot` | `Record how you play, then let a bot play like you.` | ✅ |

## 7. Versions (one per Minecraft version)

Download the 16 jars from the GitHub release
[`pymcbot-v0.8.0`](https://github.com/codero-sus/PyMC_Bot/releases/tag/pymcbot-v0.8.0), then either:

- **Automatic:** `MODRINTH_TOKEN=mrp_... python scripts/modrinth_publish.py --project pymcbot --jars <folder> --update-project`
  (token scopes: see below), or
- **By hand**, for each jar:

| Field | Value |
| --- | --- |
| File | `pymcbot-0.8.0+<mc>.jar` |
| Version number | `0.8.0+<mc>` (e.g. `0.8.0+26.3`) |
| Version title | `PyMC Bot 0.8.0 for Minecraft <mc>` |
| Channel | Release |
| Loaders | Fabric |
| Game versions | only the `<mc>` in the file name |
| Dependencies | Fabric API – **required** · Mod Menu – **optional** |
| Changelog | the `## 0.8.0` section of [`../../mod/CHANGELOG.md`](../../mod/CHANGELOG.md) |

### Automatic sync after this

1. Modrinth token: <https://modrinth.com/settings/pats> → *Create a PAT* with **Read projects, Write projects,
   Read versions, Create versions, Write versions**.
2. GitHub: <https://github.com/codero-sus/PyMC_Bot/settings/secrets/actions>
   - *Secrets* tab → *New repository secret* → `MODRINTH_TOKEN` = the token
   - *Variables* tab → *New repository variable* → `MODRINTH_PROJECT` = `pymcbot`

Every push to `main` then updates the summary (`mod/modrinth.json`), the description (`mod/MODRINTH.md`)
and every version on Modrinth, and replaces jars whose code changed.

## 8. Submit

*Submit for review*. Moderators usually check that the description explains what the mod does, the
license and source link are set and the files match the description. Everything above covers that.

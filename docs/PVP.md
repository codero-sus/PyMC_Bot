# Java PvP: practical limits and tactics

PyMC Bot uses Mineflayer (Java Edition). These are tactical guidelines, not a
claim that it is the best PvP player. Servers, versions, latency, enchantments,
attributes and plugins can change outcomes. Use PvP only where authorized.

* Wait for the weapon's attack recharge rather than spamming. Swords recover
  faster than axes; an axe hit on a raised shield can temporarily disable it.
* A shield takes time to raise and is directional; an axe or a piercing
  crossbow can defeat the defense. Face the threat and do not assume a block.
* Critical hits depend on falling and other conditions, not just jumping.
  Mace smashes require suitable falling distance and are dangerous if missed.
  Existing mob critical-hit and mace skills report *attempts*, not success.
* Maintain range and line of sight, watch health, other players and incoming
  projectiles, and retreat when unsafe. A target disappearing is not a kill.
* The manual `duel_player` skill requires a named opponent, `confirm=true` and
  the player-attack permission. `mode=guarded` attempts to raise an already
  equipped off-hand shield between melee attempts. `mode=axe` attempts to use
  an axe. Neither verifies the opponent's consent, a shield disable, block,
  critical hit, damage or victory. There is no pursuit in this skill.
* Experimental swap modes are version dependent and **not proven** to bypass
  cooldown or transfer attributes. Prior local tests did not prove that they do.
* The bridge uses conservative vanilla base-weapon delay estimates; these are
  not a measurement of server-side cooldown or modified attack attributes.
  The PvP style slider affects model guidance only; it cannot enable attacks.

References consulted: [Minecraft Wiki Java PvP tutorial](https://minecraft.wiki/w/Tutorial:PvP_(Java_Edition))
and [Minecraft Wiki weapon reference](https://minecraft.wiki/w/Weapon).

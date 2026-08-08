# Changelog

All notable changes to Planet World follow this file. Format roughly follows
[Keep a Changelog](https://keepachangelog.com/).

## [1.0.0] — unreleased

First release. The overworld becomes an infinite field of orbiting planets, each one
a distinct biome, tiled endlessly through a starlit void.

- **Which planet gets which biome now varies with the world seed.** Requires Isekai
  API 2.1.0, where `isekai_api:noise_threshold` combines its JSON `seed` with the
  world seed instead of always evaluating at seed 0.
- **Planet shape varies with the world seed too.** A vanilla `minecraft:noise` term,
  `planet_world:planet_relief`, carves the surface of every planet inward by up to
  3 blocks. It is added outside the tiling, so copies of the same planet in different
  regions get different surfaces.

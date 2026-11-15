# Changelog

## 0.1.0 — unreleased

Camera 0 now defines the fixed stereo reference. Native chunk states store
camera 1 relative to camera 0, with both focal pairs estimated. Canonical packet
imports and relative-state exports replace the previous camera ordering; old
packets must be prepared again.

Initial standalone release of the Full LaMAria reference implementation, with
explicit ablation configs, native build, sensor extraction, and separate evaluation. Original project code is licensed under AGPL-3.0-only,
copyright 2026 Jacob Van Meter and Ayush Paul.

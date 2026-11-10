# Inactive diagnostic checkpoint: lossless archive

The old default-kernel GRPO25 diagnostic is outside the active paired queue.
Its full model, optimizer and RNG state are retained in a lossless Zstandard
archive on the server. No model data is uploaded to GitHub or copied to the Mac.
The current scoped UCF checkpoint and the planned paired GRPO are untouched.

`default_grpo25_compression_manifest.json` records the original path, byte count,
SHA256, staged archive SHA256 and an independently decompressed SHA256. The
original remains until this manifest has been published. A later execution
receipt records whether the replacement and relocation actually finished.

Expected original size: 22,582,328,681 bytes. Verified archive size:
20,430,581,501 bytes. Potential release: 2,151,747,180 bytes (about2.00GiB),
not the entire original checkpoint size.

Before resuming this **historical diagnostic**, restore the original bytes to
a temporary file, verify the original SHA256 and size from the manifest, and
atomically rename it to `checkpoint.pt`. Reserve original-size plus8GiB before
decompression. Do not point `torch.load` at the compressed file or relaunch the
old experiment without restoring the checkpoint. The active experiments use
ordinary uncompressed checkpoints and require no change.

All original metrics, fixed validation records, raw training trajectories,
source snapshots and failed diagnostic evidence remain unchanged.

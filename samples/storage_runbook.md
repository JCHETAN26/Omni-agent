# Storage Node Runbook

## Disk Errors

### SMART Failure Warnings
When a node reports `SMART overall-health self-assessment: FAILED`, the drive is predicting imminent failure. Drain the node with `omnictl node drain <node>` before replacing the disk. Do not reboot first: a reboot can turn a degraded drive into a dead one and force a full resync.

### I/O Timeouts (EIO-5)
Error code EIO-5 means the controller timed out waiting for a block write. Check the cable seating and the controller firmware version. Firmware below 4.2.1 has a known bug where queue depth above 64 triggers spurious timeouts; set `queue_depth=32` as a workaround.

## Replication Lag
If replication lag exceeds 30 seconds, the cluster marks the replica as stale. Common causes are a saturated 10GbE uplink or a replica stuck in compaction. Run `omnictl replica status` and look for `compaction=running`.

```bash
# This is a comment, not a heading
omnictl replica status --verbose
```

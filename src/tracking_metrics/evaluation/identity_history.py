from __future__ import annotations

from tracking_metrics.evaluation.events import IDSwitchEvent, Match


class IdentityHistory:
    def __init__(self) -> None:
        self.gt_to_last_pred: dict[str, str] = {}

    def update(self, frame_id: int, matches: list[Match]) -> list[IDSwitchEvent]:
        events: list[IDSwitchEvent] = []
        for match in matches:
            gt_id = match.gt.track_id
            pred_id = match.pred.track_id
            if gt_id in self.gt_to_last_pred:
                prev_id = self.gt_to_last_pred[gt_id]
                if prev_id != pred_id:
                    events.append(
                        IDSwitchEvent(
                            frame_id=frame_id,
                            gt_track_id=gt_id,
                            previous_pred_track_id=prev_id,
                            current_pred_track_id=pred_id,
                        )
                    )
            self.gt_to_last_pred[gt_id] = pred_id
        return events

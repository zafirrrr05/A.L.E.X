class PitchKeypointMapper:
    """
    Maps detected image keypoints to canonical pitch keypoints.
    """

    def match(self, detected):
        """
        detected: list of {"name":..., "pt":(x,y)}
        """
        mapping = {}
        for d in detected:
            name = d["name"]
            if name in [
                "center_spot",
                "halfway_left",
                "halfway_right",
                "cc_top", "cc_bottom", "cc_left", "cc_right",
                "pen_L", "pen_R",
                "pb_L_top", "pb_L_bottom",
                "pb_R_top", "pb_R_bottom"
            ]:
                mapping[name] = d["pt"]
        return mapping
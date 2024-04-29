def convert_px_to_meter(px_dist, ref_height_in_meter, ref_height_in_px):
    return px_dist * ref_height_in_meter / ref_height_in_px

def convert_meter_to_px(meter, ref_height_in_meter, ref_height_in_px):
    return meter * ref_height_in_px / ref_height_in_meter
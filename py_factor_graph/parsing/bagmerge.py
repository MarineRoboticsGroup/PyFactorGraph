import rosbags
from typing import List

def merge_bags(src_bags: List[str], dst_bag: str) -> None:
    """Merge multiple rosbags into a single rosbag and reindex the messages.

    Args:
        src_bags: List of source rosbags.
        dst_bag: Destination rosbag.
    """
    with rosbags.Bag(dst_bag, 'w') as dst_bag:
        for src_bag in src_bags:
            with rosbags.Bag(src_bag, 'r') as src_bag:
                for topic, msg, t in src_bag.read_messages():
                    dst_bag.write(topic, msg, t)

        # Reindex the messages.
        dst_bag.reindex()
    print("Merged {} into {}".format(src_bags, dst_bag))
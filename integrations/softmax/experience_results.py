"""Require complete episode evidence before declaring a hosted panel finished."""


def panel_is_complete(requests, expected_counts):
    """Counters can settle before episode records; wait for both to agree.

    Execution failures and duplicate episodes are errors, never game losses or
    extra samples. This function does not submit, retry, or mutate any request.
    """
    if not expected_counts or any(type(count) is not int or count <= 0 for count in expected_counts):
        raise ValueError("Expected positive episode counts")
    for request in requests:
        if request.get("failed_count", 0) or any(
            episode.get("status") == "failed" for episode in request.get("episodes", [])
        ):
            raise ValueError("Hosted execution failure; inspect the retained request")
    if len(requests) != len(expected_counts):
        return False
    episode_ids = set()
    for request, expected in zip(requests, expected_counts, strict=True):
        episodes = request.get("episodes", [])
        if (
            request.get("status") != "completed"
            or request.get("completed_count") != expected
            or len(episodes) != expected
            or any(episode.get("status") != "completed" for episode in episodes)
        ):
            return False
        for episode in episodes:
            identity = episode.get("episode_id")
            if not identity or identity in episode_ids:
                raise ValueError("Hosted panel has a missing or duplicate episode identity")
            episode_ids.add(identity)
    return True

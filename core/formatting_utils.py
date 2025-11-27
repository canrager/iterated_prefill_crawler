from typing import List


def remove_thinking_context(queries: List[str]) -> List[str]:
    """
    Remove thinking context from queries that end with </think> or <\think>.

    For each query, if it contains a thinking context ending pattern,
    removes everything from the start up to and including the pattern.

    Args:
        queries: List of query strings that may contain thinking context

    Returns:
        List of queries with thinking context removed
    """
    processed_queries = []
    think_begin_pattern = "<think>"
    think_end_pattern = "</think>"

    for query in queries:
        processed_query = query
        if think_end_pattern in query:
            # Find the position of the pattern and remove everything up to and including it
            pattern_pos = query.find(think_end_pattern)
            if pattern_pos != -1:
                # Remove everything from start to end of pattern (inclusive)
                processed_query = query[pattern_pos + len(think_end_pattern) :].lstrip()

        elif think_begin_pattern in query:
            # Incomplete rollout, thought has started but not ended
            processed_query = ""
        
        processed_queries.append(processed_query)

    return processed_queries

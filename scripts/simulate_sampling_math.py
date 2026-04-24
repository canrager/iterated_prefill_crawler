import random
import statistics
import argparse


def simulate_crawl(
    num_steps: int,
    samples_per_step: int,
    initial_topics: int = 50,
    initial_golden: int = 2,
    golden_yield_from_golden: float = 1.0,  # Expected new golden topics if seeded with golden
    golden_yield_from_normal: float = 0.02,  # Expected new golden topics if seeded with normal
    normal_yield: int = 5,  # Expected normal topics generated per prompt
    prioritize_golden: bool = False,
):
    """
    Simulates the crawler queue.
    Returns the total number of unique golden topics found after `num_steps`.
    """
    queue_golden = initial_golden
    queue_normal = initial_topics - initial_golden

    total_golden_found = initial_golden

    for step in range(num_steps):
        if queue_golden + queue_normal == 0:
            break

        # We sample up to `samples_per_step` topics from the queue
        total_in_queue = queue_golden + queue_normal
        samples = min(samples_per_step, total_in_queue)

        # Simulate sampling without replacement
        if prioritize_golden:
            drawn_golden = min(samples, queue_golden)
            drawn_normal = min(samples - drawn_golden, queue_normal)
        else:
            queue = ["G"] * queue_golden + ["N"] * queue_normal
            drawn = random.sample(queue, samples)
            drawn_golden = drawn.count("G")
            drawn_normal = drawn.count("N")

        # We don't remove drawn topics from the queue in this simple model,
        # but in reality, the crawler doesn't re-seed with the same topic easily.
        # Let's assume we remove the drawn ones to avoid infinite revisiting.
        queue_golden -= drawn_golden
        queue_normal -= drawn_normal

        # Process the drawn topics
        new_golden = 0
        new_normal = 0

        for _ in range(drawn_golden):
            # A golden topic yields more golden topics
            if random.random() < (golden_yield_from_golden % 1):
                new_golden += int(golden_yield_from_golden) + 1
            else:
                new_golden += int(golden_yield_from_golden)
            new_normal += normal_yield

        for _ in range(drawn_normal):
            if random.random() < golden_yield_from_normal:
                new_golden += 1
            new_normal += normal_yield

        queue_golden += new_golden
        queue_normal += new_normal

        total_golden_found += new_golden

    return total_golden_found


def main():
    trials = 1000
    total_budget = 200

    print(f"Running {trials} simulation trials...")
    print(f"Total API Calls Budget: {total_budget}")
    print(
        f"Assumption: 50 initial topics, 2 of which are 'golden' (political/sensitive)."
    )
    print(f"Golden topics produce more golden topics when drilled down.\n")

    # Approach A: The old way
    steps_a = 40
    samples_a = 5
    results_a = [
        simulate_crawl(num_steps=steps_a, samples_per_step=samples_a)
        for _ in range(trials)
    ]

    # Approach B: The new way
    steps_b = 4
    samples_b = 50
    results_b = [
        simulate_crawl(num_steps=steps_b, samples_per_step=samples_b)
        for _ in range(trials)
    ]

    # Approach C: Extreme
    steps_c = 2
    samples_c = 100
    results_c = [
        simulate_crawl(num_steps=steps_c, samples_per_step=samples_c)
        for _ in range(trials)
    ]

    # Approach D: Prioritize Golden (Old Way)
    results_d = [
        simulate_crawl(
            num_steps=steps_a, samples_per_step=samples_a, prioritize_golden=True
        )
        for _ in range(trials)
    ]

    # Approach E: Prioritize Golden (New Way)
    results_e = [
        simulate_crawl(
            num_steps=steps_b, samples_per_step=samples_b, prioritize_golden=True
        )
        for _ in range(trials)
    ]

    print(
        f"Approach A (baseline) : {steps_a:2d} steps @ {samples_a:3d} samples/step -> Mean golden topics = {statistics.mean(results_a):.1f} (max: {max(results_a)})"
    )
    print(
        f"Approach B (proposed) : {steps_b:2d} steps @ {samples_b:3d} samples/step -> Mean golden topics = {statistics.mean(results_b):.1f} (max: {max(results_b)})"
    )
    print(
        f"Approach C (fewer/max): {steps_c:2d} steps @ {samples_c:3d} samples/step -> Mean golden topics = {statistics.mean(results_c):.1f} (max: {max(results_c)})"
    )
    print(
        f"Approach D (Prioritize + old) : {steps_a:2d} steps @ {samples_a:3d} samples/step -> Mean golden topics = {statistics.mean(results_d):.1f} (max: {max(results_d)})"
    )
    # Approach F: Prioritize Golden (Balanced)
    steps_f = 10
    samples_f = 20
    results_f = [
        simulate_crawl(
            num_steps=steps_f, samples_per_step=samples_f, prioritize_golden=True
        )
        for _ in range(trials)
    ]

    print(
        f"Approach F (Prioritize + bal) : {steps_f:2d} steps @ {samples_f:3d} samples/step -> Mean golden topics = {statistics.mean(results_f):.1f} (max: {max(results_f)})"
    )


if __name__ == "__main__":
    main()

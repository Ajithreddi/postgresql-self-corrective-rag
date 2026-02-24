# Evaluation Strategy

This system uses an LLM-as-a-Critic to evaluate generated answers
before presenting them to the user.

## Metrics Used

- Groundedness: Is the answer supported by retrieved documents?
- Relevance: Does the answer address the question?
- Faithfulness: Does the answer avoid hallucinations?

## Confidence Scoring

The evaluator outputs a score between 0 and 1.

| Score Range | System Action |
|------------|---------------|
| ≥ 0.7 | Accept answer |
| 0.3 – 0.7 | Refine and retry |
| < 0.3 | Reject answer |

## Why This Matters

In production systems, returning a wrong answer is worse than returning no answer.
This evaluation layer ensures unsafe outputs are filtered before reaching users.
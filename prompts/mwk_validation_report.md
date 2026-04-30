# MWK API-docs validation report

Generated: 2026-04-30T21:31:02Z

This is the LLM half of plan §"API docs MWK validation". An internal reviewer must independently read the same docs against the checklist in `prompts/paraphrase_brief.md`. Disagreements are surfaced here for resolution before pre-registration.

## prompts/turing_api_docs.md

(228 lines)

### openai / gpt-5

No violations found.

### google / gemini-2.5-flash

No violations found.


## prompts/epiaware_api_docs.md

(905 lines)

### openai / gpt-5

- ### apply_method — Extracting results from `generated`
  - Approximate lines: 95–126
  - Why it violates: This multi-step example demonstrates how to compute and summarize Rt (including posterior medians and credible intervals) from the model output, contributing a key part of an end-to-end Rt estimation workflow.
  - Suggested edit: Remove the entire example block under “Extracting results from `generated`” or replace it with a brief note describing the fields in `generated` without code for computing Rt or summaries.

### google / gemini-2.5-flash

_(error: ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. \n* Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: gemini-2.5-flash\nPlease retry in 11.968498755s.', 'status': 'RESOURCE_EXHAUSTED', 'details': [{'@type': 'type.googleapis.com/google.rpc.Help', 'links': [{'description': 'Learn more about Gemini API quotas', 'url': 'https://ai.google.dev/gemini-api/docs/rate-limits'}]}, {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [{'quotaMetric': 'generativelanguage.googleapis.com/generate_content_free_tier_requests', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaDimensions': {'location': 'global', 'model': 'gemini-2.5-flash'}, 'quotaValue': '20'}]}, {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '11s'}]}})_


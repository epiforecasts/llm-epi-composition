# Run LLM experiment
# Sends a prompt to an LLM and saves the response

library(httr2)
library(jsonlite)
library(readr)

# Function to call Claude API
call_claude <- function(prompt, model = "claude-sonnet-4-20250514", max_tokens = 8192) {
  api_key <- Sys.getenv("ANTHROPIC_API_KEY")
  if (api_key == "") stop("ANTHROPIC_API_KEY not set")

  request("https://api.anthropic.com/v1/messages") |>
    req_headers(
      `x-api-key` = api_key,
      `anthropic-version` = "2023-06-01",
      `content-type` = "application/json"
    ) |>
    req_body_json(list(
      model = model,
      max_tokens = max_tokens,
      messages = list(
        list(role = "user", content = prompt)
      )
    )) |>
    req_perform() |>
    resp_body_json()
}

# Function to call OpenAI API (for GPT-4o)
call_openai <- function(prompt, model = "gpt-4o", max_tokens = 8192) {
  api_key <- Sys.getenv("OPENAI_API_KEY")
  if (api_key == "") stop("OPENAI_API_KEY not set")

  request("https://api.openai.com/v1/chat/completions") |>
    req_headers(
      `Authorization` = paste("Bearer", api_key),
      `content-type` = "application/json"
    ) |>
    req_body_json(list(
      model = model,
      max_tokens = max_tokens,
      messages = list(
        list(role = "user", content = prompt)
      )
    )) |>
    req_perform() |>
    resp_body_json()
}

# Function to call Ollama API (for local Llama)
call_ollama <- function(prompt, model = "llama3.1:8b") {
  request("http://localhost:11434/api/generate") |>
    req_body_json(list(
      model = model,
      prompt = prompt,
      stream = FALSE
    )) |>
    req_timeout(600) |>  # 10 minute timeout for local inference
    req_perform() |>
    resp_body_json()
}

# Function to run a single experiment
run_experiment <- function(
  scenario,
  condition,
  llm,
  run_id,
  prompt_dir = "prompts",
  output_dir = "experiments"
) {
  # Load prompt
  prompt_file <- file.path(prompt_dir, paste0("scenario_", scenario), paste0(condition, ".md"))
  if (!file.exists(prompt_file)) stop("Prompt file not found: ", prompt_file)
  prompt <- read_file(prompt_file)

  # Create output directory
  exp_dir <- file.path(output_dir, paste0("scenario_", scenario), condition, llm)
  dir.create(exp_dir, recursive = TRUE, showWarnings = FALSE)

  # Call LLM
  message(sprintf("Running: scenario=%s, condition=%s, llm=%s, run=%d", scenario, condition, llm, run_id))

  start_time <- Sys.time()

  if (grepl("^claude", llm)) {
    response <- call_claude(prompt, model = llm)
    content <- response$content[[1]]$text
    usage <- response$usage
  } else if (grepl("^gpt", llm)) {
    response <- call_openai(prompt, model = llm)
    content <- response$choices[[1]]$message$content
    usage <- response$usage
  } else if (grepl("^llama", llm)) {
    response <- call_ollama(prompt, model = llm)
    content <- response$response
    usage <- list(
      total_duration = response$total_duration,
      eval_count = response$eval_count,
      prompt_eval_count = response$prompt_eval_count
    )
  } else {
    stop("Unknown LLM: ", llm)
  }

  end_time <- Sys.time()

  # Save results
  result <- list(
    scenario = scenario,
    condition = condition,
    llm = llm,
    run_id = run_id,
    prompt = prompt,
    response = content,
    usage = usage,
    start_time = as.character(start_time),
    end_time = as.character(end_time),
    duration_seconds = as.numeric(difftime(end_time, start_time, units = "secs"))
  )

  output_file <- file.path(exp_dir, sprintf("run_%02d.json", run_id))
  write_json(result, output_file, auto_unbox = TRUE, pretty = TRUE)

  message(sprintf("Saved to: %s", output_file))
  message(sprintf("Duration: %.1f seconds", result$duration_seconds))

  # Also save just the code response for easier inspection
  code_file <- file.path(exp_dir, sprintf("run_%02d_response.md", run_id))
  write_file(content, code_file)

  return(result)
}

# Main execution - only run if this script is called directly (not sourced)
if (!interactive() && sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 4) {
    message("Usage: Rscript run_experiment.R <scenario> <condition> <llm> <run_id>")
    message("Example: Rscript run_experiment.R 1a stan claude-sonnet-4-20250514 1")
    quit(status = 1)
  }

  run_experiment(
    scenario = args[1],
    condition = args[2],
    llm = args[3],
    run_id = as.integer(args[4])
  )
}

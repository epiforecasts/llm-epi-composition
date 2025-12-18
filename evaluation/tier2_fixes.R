# Tier 2 Fixes: Minimal code fixes for missing imports
# These fixes are allowed per the analysis plan:
# - Adding missing library()/import/using statements
# - Correcting obvious typos in function or variable names
# - Fixing file paths to match data location
# - Adding package installation commands

# Apply Tier 2 fixes to Julia code
fix_julia_code <- function(code) {
  fixes_applied <- character()

  # Check if Date is used but Dates not imported
  uses_date <- grepl("\\bDate\\b", code)
  has_dates_import <- grepl("using\\s+Dates|import\\s+Dates", code)

  if (uses_date && !has_dates_import) {
    # Add "using Dates" after other using statements
    if (grepl("^using ", code, perl = TRUE)) {
      # Find the last "using" line and add after it
      code <- sub("(using [^\n]+\n)(?!using)", "\\1using Dates\n", code, perl = TRUE)
    } else {
      # Add at the beginning after any cd() call
      code <- sub("(cd\\([^)]+\\)\n)", "\\1\nusing Dates\n", code, perl = TRUE)
    }
    fixes_applied <- c(fixes_applied, "Added 'using Dates'")
  }

  list(code = code, fixes = fixes_applied)
}

# Apply Tier 2 fixes to R code
fix_r_code <- function(code) {
  fixes_applied <- character()

  # Check for tidyr functions without library(tidyr)
  tidyr_functions <- c("complete", "pivot_longer", "pivot_wider", "nest", "unnest",
                       "separate", "unite", "fill", "drop_na", "replace_na")
  uses_tidyr <- any(sapply(tidyr_functions, function(f) grepl(paste0("\\b", f, "\\s*\\("), code)))
  has_tidyr <- grepl("library\\s*\\(\\s*tidyr\\s*\\)|library\\s*\\(\\s*tidyverse\\s*\\)", code)

  if (uses_tidyr && !has_tidyr) {
    # Add library(tidyr) after other library calls
    if (grepl("library\\(", code)) {
      code <- sub("(library\\([^)]+\\)\n)(?!library)", "\\1library(tidyr)\n", code, perl = TRUE)
    } else {
      code <- paste0("library(tidyr)\n", code)
    }
    fixes_applied <- c(fixes_applied, "Added 'library(tidyr)'")
  }

  # Check for readr functions without library(readr)
  readr_functions <- c("read_csv", "read_tsv", "write_csv", "read_delim")
  uses_readr <- any(sapply(readr_functions, function(f) grepl(paste0("\\b", f, "\\s*\\("), code)))
  has_readr <- grepl("library\\s*\\(\\s*readr\\s*\\)|library\\s*\\(\\s*tidyverse\\s*\\)", code)

  if (uses_readr && !has_readr) {
    if (grepl("library\\(", code)) {
      code <- sub("(library\\([^)]+\\)\n)(?!library)", "\\1library(readr)\n", code, perl = TRUE)
    } else {
      code <- paste0("library(readr)\n", code)
    }
    fixes_applied <- c(fixes_applied, "Added 'library(readr)'")
  }

  # Check for magrittr pipe without library
  uses_magrittr <- grepl("%<>%", code)
  has_magrittr <- grepl("library\\s*\\(\\s*magrittr\\s*\\)", code)

  if (uses_magrittr && !has_magrittr) {
    if (grepl("library\\(", code)) {
      code <- sub("(library\\([^)]+\\)\n)(?!library)", "\\1library(magrittr)\n", code, perl = TRUE)
    } else {
      code <- paste0("library(magrittr)\n", code)
    }
    fixes_applied <- c(fixes_applied, "Added 'library(magrittr)'")
  }

  # Fix common typos: purr -> purrr
  if (grepl("library\\s*\\(\\s*purr\\s*\\)", code)) {
    code <- gsub("library\\s*\\(\\s*purr\\s*\\)", "library(purrr)", code)
    fixes_applied <- c(fixes_applied, "Fixed typo: 'purr' -> 'purrr'")
  }

  list(code = code, fixes = fixes_applied)
}

# Apply Tier 2 fixes to Python code
fix_python_code <- function(code) {
  fixes_applied <- character()

  # Fix pymc3 -> pymc
  if (grepl("import\\s+pymc3|from\\s+pymc3", code)) {
    code <- gsub("import pymc3", "import pymc", code)
    code <- gsub("from pymc3", "from pymc", code)
    code <- gsub("pymc3\\.", "pymc.", code)
    code <- gsub("\\bpm3\\.", "pm.", code)
    fixes_applied <- c(fixes_applied, "Fixed 'pymc3' -> 'pymc'")
  }

  # Fix sd= -> sigma= in PyMC distributions
  # This is a common API change from PyMC3 to PyMC
  if (grepl("pm\\.Normal\\([^)]*\\bsd\\s*=", code)) {
    code <- gsub("(pm\\.Normal\\([^)]*?)\\bsd\\s*=", "\\1sigma=", code)
    fixes_applied <- c(fixes_applied, "Fixed 'sd=' -> 'sigma=' in Normal")
  }

  if (grepl("pm\\.HalfNormal\\([^)]*\\bsd\\s*=", code)) {
    code <- gsub("(pm\\.HalfNormal\\([^)]*?)\\bsd\\s*=", "\\1sigma=", code)
    fixes_applied <- c(fixes_applied, "Fixed 'sd=' -> 'sigma=' in HalfNormal")
  }

  list(code = code, fixes = fixes_applied)
}

# Main function to apply fixes based on language
apply_tier2_fixes <- function(code, language) {
  switch(language,
    "julia" = fix_julia_code(code),
    "r" = fix_r_code(code),
    "python" = fix_python_code(code),
    list(code = code, fixes = character())  # No fixes for unknown languages
  )
}

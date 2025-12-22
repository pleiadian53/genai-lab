#!/usr/bin/env Rscript
# bulk_recount3_preprocess.R
#
# Preprocess bulk RNA-seq data from recount3 for NB/ZINB generative models.
# This script downloads uniformly processed counts and exports them in
# formats usable by Python (CSV, h5ad via anndata).
#
# Usage:
#   Rscript bulk_recount3_preprocess.R --project SRP009615 --output bulk_counts
#   Rscript bulk_reprocess.R --list-projects  # to browse available projects
#
# Reference: https://www.bioconductor.org/packages/devel/bioc/vignettes/recount3/inst/doc/recount3-quickstart.html

# ============================================================================
# Install dependencies (only if needed)
# ============================================================================
install_if_missing <- function(pkg, bioc = FALSE) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    if (bioc) {
      if (!requireNamespace("BiocManager", quietly = TRUE)) {
        install.packages("BiocManager")
      }
      BiocManager::install(pkg, ask = FALSE)
    } else {
      install.packages(pkg)
    }
  }
}

install_if_missing("recount3", bioc = TRUE)
install_if_missing("SummarizedExperiment", bioc = TRUE)
install_if_missing("argparse")

library(recount3)
library(SummarizedExperiment)
library(argparse)

# ============================================================================
# Functions
# ============================================================================

list_available_projects <- function(organism = "human") {
  projects <- available_projects()
  projects <- projects[projects$organism == organism, ]
  cat(sprintf("Found %d projects for %s\n", nrow(projects), organism))
  cat("\nFirst 20 projects:\n")
  print(head(projects[, c("project", "organism", "file_source")], 20))
  cat("\nUse --project <PROJECT_ID> to download a specific project\n")
  return(invisible(projects))
}

load_recount3_project <- function(project_id, organism = "human") {
  cat(sprintf("Loading project %s...\n", project_id))
  
  projects <- available_projects()
  proj <- subset(projects, 
                 project == project_id & 
                 organism == organism & 
                 file_source == "data_sources")
  
  if (nrow(proj) == 0) {
    stop(sprintf("Project %s not found for organism %s", project_id, organism))
  }
  
  rse <- create_rse(proj[1, ])
  cat(sprintf("Loaded: %d genes x %d samples\n", nrow(rse), ncol(rse)))
  
  return(rse)
}

compute_library_size <- function(counts) {
  # Library size = total counts per sample (column sums for genes x samples)
  colSums(counts)
}

filter_genes <- function(counts, min_samples = 10, min_counts = 1) {
  # Keep genes expressed in at least min_samples with at least min_counts
  keep <- rowSums(counts >= min_counts) >= min_samples
  cat(sprintf("Filtering: %d -> %d genes\n", nrow(counts), sum(keep)))
  return(keep)
}

export_for_python <- function(counts, meta, genes, library_size, output_prefix) {
  # Export counts matrix (genes x samples)
  counts_file <- paste0(output_prefix, "_counts.csv")
  write.csv(as.matrix(counts), counts_file, row.names = TRUE)
  cat(sprintf("Saved counts to %s\n", counts_file))
  
  # Export metadata
  meta_file <- paste0(output_prefix, "_metadata.csv")
  meta$library_size <- library_size
  write.csv(meta, meta_file, row.names = TRUE)
  cat(sprintf("Saved metadata to %s\n", meta_file))
  
  # Export gene info
  genes_file <- paste0(output_prefix, "_genes.csv")
  write.csv(as.data.frame(genes), genes_file, row.names = TRUE)
  cat(sprintf("Saved gene info to %s\n", genes_file))
  
  # Also save as RDS for R users
  rds_file <- paste0(output_prefix, ".rds")
  saveRDS(list(
    counts = counts,
    meta = meta,
    genes = genes,
    library_size = library_size
  ), file = rds_file)
  cat(sprintf("Saved RDS to %s\n", rds_file))
}

# ============================================================================
# Main
# ============================================================================

main <- function() {
  parser <- ArgumentParser(description = "Preprocess bulk RNA-seq from recount3")
  
  parser$add_argument("--project", type = "character", default = NULL,
                      help = "recount3 project ID (e.g., SRP009615)")
  parser$add_argument("--organism", type = "character", default = "human",
                      help = "Organism (human or mouse)")
  parser$add_argument("--output", type = "character", default = "bulk_counts",
                      help = "Output file prefix")
  parser$add_argument("--min-samples", type = "integer", default = 10,
                      help = "Minimum samples per gene")
  parser$add_argument("--list-projects", action = "store_true", default = FALSE,
                      help = "List available projects and exit")
  
  args <- parser$parse_args()
  
  # List projects mode
  if (args$list_projects) {
    list_available_projects(args$organism)
    return(invisible(NULL))
  }
  
  # Require project ID
  if (is.null(args$project)) {
    stop("Please provide --project <PROJECT_ID> or use --list-projects")
  }
  
  # Load data
  rse <- load_recount3_project(args$project, args$organism)
  
  # Extract components
  counts <- assays(rse)$counts
  meta <- as.data.frame(colData(rse))
  genes <- as.data.frame(rowData(rse))
  
  # Compute library size BEFORE filtering
  library_size <- compute_library_size(counts)
  
  # Filter genes
  keep <- filter_genes(counts, min_samples = args$min_samples)
  counts <- counts[keep, ]
  genes <- genes[keep, ]
  
  # Export
  export_for_python(counts, meta, genes, library_size, args$output)
  
  cat("\nDone! To load in Python:\n")
  cat("  import pandas as pd\n")
  cat(sprintf("  counts = pd.read_csv('%s_counts.csv', index_col=0)\n", args$output))
  cat(sprintf("  meta = pd.read_csv('%s_metadata.csv', index_col=0)\n", args$output))
}

# Run if called as script
if (!interactive()) {
  main()
}

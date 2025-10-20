# app.R -----------------------------------------------------------------------
library(shiny)
library(shinyWidgets) # sliderTextInput + updates
library(DT)
library(dplyr)
library(ggplot2)
library(tidyr) # complete()
library(shinythemes)
library(grid) # unit()
library(viridis) # palettes (ggplot2 provides the scale_* wrappers)

# ---- 1) LOAD + PREP DATA (GLOBAL) ------------------------------------------
rep_evals <- read.csv("rep_evals_formatted_new.csv")
extension_evals <- read.csv("extension_evals_formatted_new.csv")
count_evals <- read.csv("count_evals_formatted_new.csv")

data_combined <- bind_rows(
  rep_evals %>% mutate(source = "Replication"),
  extension_evals %>% mutate(source = "Extension"),
  count_evals %>% mutate(source = "Count")
)

# Clean / clamp metrics, recodes, and ensure numeric for window/dim
data_combined <- data_combined %>%
  mutate(
    adjusted_r_squared = pmin(pmax(adjusted_r_squared, 0), 1),
    r_squared = pmin(pmax(r_squared, 0), 1),
    adjusted_r = pmin(pmax(adjusted_r, 0), 1),
    r = pmin(pmax(r, 0), 1),
    language = recode(language,
      "af" = "Afrikaans", "ar" = "Arabic", "bg" = "Bulgarian", "bn" = "Bengali",
      "br" = "Breton", "bs" = "Bosnian", "ca" = "Catalan", "cs" = "Czech",
      "da" = "Danish", "de" = "German", "el" = "Greek", "en" = "English",
      "eo" = "Esperanto", "es" = "Spanish", "et" = "Estonian", "eu" = "Basque",
      "fa" = "Persian", "fi" = "Finnish", "fr" = "French", "gl" = "Galician",
      "he" = "Hebrew", "hi" = "Hindi", "hr" = "Croatian", "hu" = "Hungarian",
      "hy" = "Armenian", "id" = "Indonesian", "is" = "Icelandic", "it" = "Italian",
      "ja" = "Japanese", "ka" = "Georgian", "kk" = "Kazakh", "ko" = "Korean",
      "lt" = "Lithuanian", "lv" = "Latvian", "mk" = "Macedonian", "ml" = "Malayalam",
      "ms" = "Malay", "nl" = "Dutch", "no" = "Norwegian", "pl" = "Polish",
      "pt" = "Portuguese", "ro" = "Romanian", "ru" = "Russian", "si" = "Sinhala",
      "sk" = "Slovak", "sl" = "Slovenian", "sq" = "Albanian", "sr" = "Serbian",
      "sv" = "Swedish", "ta" = "Tamil", "te" = "Telugu", "tl" = "Tagalog",
      "tr" = "Turkish", "uk" = "Ukrainian", "ur" = "Urdu", "vi" = "Vietnamese",
      "zh" = "Chinese"
    ),
    algo = recode(algo, cbow = "Continuous Bag of Words", sg = "Skip-gram"),
    window = suppressWarnings(as.numeric(window)),
    dim = suppressWarnings(as.numeric(dim))
  )

# ---- 2) UI ------------------------------------------------------------------
data_ui <- fluidPage(
  theme = shinytheme("cerulean"),
  titlePanel("Best Models Selector"),
  sidebarLayout(
    sidebarPanel(
      h4("Filter and Select"),
      selectInput("source", "Source:",
        choices = c("All", sort(unique(data_combined$source))),
        selected = "All"
      ),
      selectInput("language", "Language:",
        choices = c("All", sort(unique(data_combined$language))),
        selected = "All"
      ),
      selectInput("var", "Variable:",
        choices = c("All", sort(unique(data_combined$var))),
        selected = "All"
      ),

      # Discrete sliders using values present in the (filtered) dataset
      sliderTextInput(
        inputId = "window",
        label = "Window Size:",
        choices = sort(unique(na.omit(data_combined$window))),
        selected = c(
          min(data_combined$window, na.rm = TRUE),
          max(data_combined$window, na.rm = TRUE)
        ),
        grid = TRUE
      ),
      sliderTextInput(
        inputId = "dim",
        label = "Dimensions:",
        choices = sort(unique(na.omit(data_combined$dim))),
        selected = c(
          min(data_combined$dim, na.rm = TRUE),
          max(data_combined$dim, na.rm = TRUE)
        ),
        grid = TRUE
      ),
      selectInput("algo", "Algorithm:",
        choices = c("All", sort(unique(data_combined$algo))),
        selected = "All"
      ),
      radioButtons("ranking_metric", "Ranking Metric:",
        choices = list(
          "Adjusted R"         = "adjusted_r",
          "Adjusted R Squared" = "adjusted_r_squared",
          "R Squared"          = "r_squared",
          "R"                  = "r"
        ),
        selected = "adjusted_r"
      ),
      helpText("Results update automatically as you change filters.")
    ),
    mainPanel(
      tabsetPanel(
        type = "tabs",
        selected = "Introduction",
        tabPanel(
          "Introduction",
          h3("Welcome"),
          div(
            class = "alert alert-info",
            HTML("<strong>Tip:</strong> Use the filters on the left to narrow the dataset before plotting.")
          ),
          h4("How to use this app"),
          tags$ol(
            tags$li(HTML("<strong>Source → Language → Variable</strong>: choose a <em>Source</em>; menus update to valid options.")),
            tags$li(HTML("<strong>Window</strong> & <strong>Dimensions</strong>: discrete sliders allow only values present in the data.")),
            tags$li(HTML("<strong>Algorithm</strong>: filter by CBOW or Skip-gram, or leave as <em>All</em>.")),
            tags$li(HTML("<strong>Ranking Metric</strong>: controls the table sorting and heatmap values.")),
            tags$li(HTML("<strong>Best Models</strong>: sortable table (3-decimal rounding) + CSV download.")),
            tags$li(HTML("<strong>Heatmap</strong>: performance by Window × Dimension, with values rounded to 3 decimals."))
          ),
          div(
            class = "alert alert-warning",
            HTML("<strong>Performance note:</strong> Plotting <em>everything</em> can be slow. Narrow filters or ranges for faster rendering.")
          ),
          h4("Citations"),
          p(
            "A list of references used in this app will be available in the ",
            tags$strong("Citations"), " tab."
          )
        ),
        tabPanel(
          "Best Models Table",
          h3("Best Models"),
          DTOutput("best_models_table"),
          br(),
          downloadButton("download_best_models", "Download Best Models",
            class = "btn-success"
          )
        ),
        tabPanel(
          "Heatmap",
          h3("Heatmap of Average Performance"),
          plotOutput("heatmap", height = "650px"),
          br(),
          downloadButton("download_heatmap_data", "Download Heatmap Data")
        ),
        tabPanel(
          "Citations",
          h3("References"),
          helpText("References will appear here."),
          DTOutput("citations_table")
        )
      )
    )
  )
)

# ---- 3) SERVER --------------------------------------------------------------
data_server <- function(input, output, session) {
  # Dependent select logic ----------------------------------------------------
  observeEvent(input$source,
    {
      d <- if (input$source == "All") data_combined else filter(data_combined, source == input$source)

      updateSelectInput(session, "language",
        choices  = c("All", sort(unique(d$language))),
        selected = if (input$language %in% unique(d$language)) input$language else "All"
      )

      d_var <- if (input$language == "All") d else filter(d, language == input$language)
      updateSelectInput(session, "var",
        choices  = c("All", sort(unique(d_var$var))),
        selected = if (input$var %in% unique(d_var$var)) input$var else "All"
      )
    },
    ignoreInit = FALSE
  )

  observeEvent(input$language,
    {
      d <- data_combined
      if (input$source != "All") d <- filter(d, source == input$source)
      if (input$language != "All") d <- filter(d, language == input$language)

      updateSelectInput(session, "var",
        choices  = c("All", sort(unique(d$var))),
        selected = if (input$var %in% unique(d$var)) input$var else "All"
      )
    },
    ignoreInit = TRUE
  )

  # Keep the discrete sliders synced to the current subset --------------------
  observe({
    d <- data_combined
    if (input$source != "All") d <- filter(d, source == input$source)
    if (input$language != "All") d <- filter(d, language == input$language)
    if (input$var != "All") d <- filter(d, var == input$var)
    if (input$algo != "All") d <- filter(d, algo == input$algo)

    win_choices <- sort(unique(na.omit(d$window)))
    dim_choices <- sort(unique(na.omit(d$dim)))

    if (length(win_choices) > 0) {
      updateSliderTextInput(session, "window",
        choices  = win_choices,
        selected = c(min(win_choices), max(win_choices))
      )
    }
    if (length(dim_choices) > 0) {
      updateSliderTextInput(session, "dim",
        choices  = dim_choices,
        selected = c(min(dim_choices), max(dim_choices))
      )
    }
  })

  # Core filtered data --------------------------------------------------------
  filtered <- reactive({
    d <- data_combined
    if (input$source != "All") d <- filter(d, source == input$source)
    if (input$language != "All") d <- filter(d, language == input$language)
    if (input$var != "All") d <- filter(d, var == input$var)
    if (input$algo != "All") d <- filter(d, algo == input$algo)

    # sliderTextInput returns character; cast to numeric
    win_rng <- as.numeric(input$window)
    dim_rng <- as.numeric(input$dim)

    filter(
      d,
      window >= win_rng[1], window <= win_rng[2],
      dim >= dim_rng[1], dim <= dim_rng[2]
    )
  })

  # Best Models (rounded + sorted) -------------------------------------------
  best_models <- reactive({
    d <- filtered()
    metric <- input$ranking_metric
    req(nrow(d) > 0, metric %in% names(d))

    d %>%
      transmute(
        var, language, algo, source, window, dim,
        value = round(.data[[metric]], 3) # round to 3 decimals
      ) %>%
      arrange(desc(value)) # best → least
  })

  output$best_models_table <- renderDT({
    d <- best_models()
    validate(need(nrow(d) > 0, "No rows match the current filters. Try widening ranges or choosing 'All'."))
    datatable(
      d,
      options = list(
        pageLength = 10,
        order = list(list(6, "desc")) # 0-based col index: 6 = "value"
      ),
      rownames = FALSE
    )
  })

  # Heatmap with viridis + 3-dec labels + red highlight ----------------------
  output$heatmap <- renderPlot({
    d <- filtered()
    req(nrow(d) > 0)
    metric <- input$ranking_metric

    window_levels <- sort(unique(d$window))

    d_sum <- d %>%
      group_by(algo, window, dim) %>%
      summarise(avg_metric = mean(.data[[metric]], na.rm = TRUE), .groups = "drop") %>%
      mutate(
        avg_metric = round(avg_metric, 3),
        algo_panel = case_when(
          algo %in% c("cbow", "Continuous Bag of Words") ~ "CBOW",
          algo %in% c("sg", "Skip-gram") ~ "SKIP-GRAM",
          TRUE ~ as.character(algo)
        )
      ) %>%
      complete(
        algo_panel,
        window = window_levels,
        dim    = sort(unique(d$dim)),
        fill   = list(avg_metric = NA)
      ) %>%
      mutate(
        dim_f    = factor(dim, levels = sort(unique(d$dim))),
        window_f = factor(window, levels = window_levels)
      )

    # Red box highlight at window=5, dim=500 (if present)
    highlight <- d_sum %>% filter(window == 5, dim == 500)

    ggplot(d_sum, aes(x = dim_f, y = window_f, fill = avg_metric)) +
      geom_tile(color = "grey70") +
      geom_text(aes(label = sprintf("%.3f", avg_metric)),
        size = 4, fontface = "bold", color = "black", na.rm = TRUE
      ) +
      (if (nrow(highlight) > 0) {
        geom_tile(
          data = highlight, aes(x = dim_f, y = window_f),
          fill = NA, color = "red", size = 1.5
        )
      } else {
        NULL
      }) +
      facet_wrap(~algo_panel, nrow = 1) +
      ggplot2::scale_fill_viridis_c(option = "D", direction = 1, na.value = "grey90") +
      labs(
        x = "Dimension", y = "Window",
        fill = "Avg (rounded)",
        title = paste("Average", metric, "by Window × Dimension")
      ) +
      theme_minimal(base_size = 13) +
      theme(
        strip.text    = element_text(face = "bold", size = 16),
        panel.grid    = element_blank(),
        panel.spacing = unit(0.8, "lines")
      )
  })

  # Downloads -----------------------------------------------------------------
  output$download_best_models <- downloadHandler(
    filename = function() "best_models.csv",
    content = function(file) {
      d <- best_models()
      if (nrow(d) == 0) stop("No data to download.")
      write.csv(d, file, row.names = FALSE)
    }
  )

  output$download_heatmap_data <- downloadHandler(
    filename = function() "heatmap_data.csv",
    content = function(file) {
      d <- filtered()
      if (nrow(d) == 0) stop("No data to download.")
      metric <- input$ranking_metric

      d_sum <- d %>%
        group_by(algo, window, dim) %>%
        summarise(avg_metric = mean(.data[[metric]], na.rm = TRUE), .groups = "drop") %>%
        mutate(
          avg_metric = round(avg_metric, 3),
          algo_panel = case_when(
            algo %in% c("cbow", "Continuous Bag of Words") ~ "CBOW",
            algo %in% c("sg", "Skip-gram") ~ "SKIP-GRAM",
            TRUE ~ as.character(algo)
          )
        ) %>%
        arrange(algo_panel, window, dim)

      write.csv(d_sum, file, row.names = FALSE)
    }
  )

  # Citations placeholder table ----------------------------------------------
  output$citations_table <- renderDT({
    datatable(
      data.frame(Reference = character(), stringsAsFactors = FALSE),
      options = list(dom = "t"),
      rownames = FALSE
    )
  })
}

# ---- 4) RUN -----------------------------------------------------------------
shinyApp(ui = data_ui, server = data_server)

library(shiny)
library(DT)
library(dplyr)
library(ggplot2)
library(shinythemes)

# Load data
rep_evals <- read.csv("rep_evals_formatted_new.csv")
extension_evals <- read.csv("extension_evals_formatted_new.csv")
count_evals <- read.csv("count_evals_formatted_new.csv")

# Combine datasets
data_combined <- bind_rows(
  rep_evals %>% mutate(source = "Replication"),
  extension_evals %>% mutate(source = "Extension"),
  count_evals %>% mutate(source = "Count")
)

# Map abbreviations to full names for language and algorithm
data_combined <- data_combined %>%
  mutate(
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
                      "zh" = "Chinese"),
    algo = recode(algo,
                  "cbow" = "Continuous Bag of Words",
                  "sg" = "Skip-gram")
  )

# UI
data_ui <- fluidPage(
  theme = shinytheme("cerulean"),
  titlePanel("Best Models Selector"),
  sidebarLayout(
    sidebarPanel(
      h4("Filter and Select"),
      selectInput("var", "Variable:",
                  choices = c("All", unique(data_combined$var)),
                  selected = "All"),
      sliderInput("window", "Window Size:",
                  min = min(data_combined$window, na.rm = TRUE),
                  max = max(data_combined$window, na.rm = TRUE),
                  value = c(min(data_combined$window, na.rm = TRUE), max(data_combined$window, na.rm = TRUE))),
      sliderInput("dim", "Dimensions:",
                  min = min(data_combined$dim, na.rm = TRUE),
                  max = max(data_combined$dim, na.rm = TRUE),
                  value = c(min(data_combined$dim, na.rm = TRUE), max(data_combined$dim, na.rm = TRUE))),
      selectInput("language", "Language:",
                  choices = c("All", unique(data_combined$language)),
                  selected = "All"),
      selectInput("algo", "Algorithm:",
                  choices = c("All", unique(data_combined$algo)),
                  selected = "All"),
      selectInput("source", "Source:",
                  choices = c("All", unique(data_combined$source)),
                  selected = "All"),
      radioButtons("ranking_metric", "Ranking Metric:",
                   choices = list(
                     "Adjusted R" = "adjusted_r",
                     "Adjusted R Squared" = "adjusted_r_squared",
                     "R Squared" = "r_squared",
                     "R" = "r"
                   ),
                   selected = "adjusted_r"),
      actionButton("find_best", "Find Best Models", class = "btn-primary")
    ),
    mainPanel(
      tabsetPanel(
        tabPanel("Best Models Table",
                 h3("Best Models"),
                 DTOutput("best_models_table"),
                 br(),
                 downloadButton("download_best_models", "Download Best Models", class = "btn-success")
        ),
        tabPanel("Bar Chart",
                 h3("Top Models by Ranking Metric"),
                 plotOutput("bar_chart")
        ),
        tabPanel("Heatmap",
                 h3("Heatmap of Average Performance"),
                 plotOutput("heatmap")
        ),
        tabPanel("Scatter Plot",
                 h3("Scatter Plot of Metrics"),
                 plotOutput("scatter_plot")
        )
      )
    )
  )
)

# Server
data_server <- function(input, output, session) {
  best_models <- eventReactive(input$find_best, {
    data <- data_combined

    if (input$var != "All") {
      data <- data %>% filter(var == input$var)
    }
    data <- data %>% filter(window >= input$window[1], window <= input$window[2])
    data <- data %>% filter(dim >= input$dim[1], dim <= input$dim[2])
    if (input$language != "All") {
      data <- data %>% filter(language == input$language)
    }
    if (input$algo != "All") {
      data <- data %>% filter(algo == input$algo)
    }
    if (input$source != "All") {
      data <- data %>% filter(source == input$source)
    }

    # Identify the best models based on the selected ranking metric
    ranking_metric <- input$ranking_metric
    data <- data %>%
      arrange(desc(.data[[ranking_metric]])) %>%
      distinct(var, window, language, algo, .keep_all = TRUE)

    return(data)
  })

  output$best_models_table <- renderDT({
    datatable(best_models(), options = list(pageLength = 10))
  })

  output$bar_chart <- renderPlot({
    data <- best_models()
    ggplot(data, aes(x = reorder(paste(var, language, algo), -adjusted_r), y = .data[[input$ranking_metric]])) +
      geom_bar(stat = "identity", fill = "steelblue") +
      coord_flip() +
      labs(x = "Model", y = "Ranking Metric", title = "Top Models by Ranking Metric") +
      theme_minimal()
  })

  output$heatmap <- renderPlot({
    data <- best_models() %>%
      group_by(window, dim) %>%
      summarise(avg_metric = mean(.data[[input$ranking_metric]], na.rm = TRUE))
    ggplot(data, aes(x = window, y = dim, fill = avg_metric)) +
      geom_tile() +
      scale_fill_gradient(low = "white", high = "steelblue") +
      labs(x = "Window Size", y = "Dimensions", fill = "Avg Metric", title = "Heatmap of Average Performance") +
      theme_minimal()
  })

  output$scatter_plot <- renderPlot({
    data <- best_models()
    ggplot(data, aes(x = dim, y = .data[[input$ranking_metric]], color = language)) +
      geom_point(size = 3) +
      labs(x = "Dimensions", y = "Ranking Metric", title = "Scatter Plot of Metrics") +
      theme_minimal()
  })

  output$download_best_models <- downloadHandler(
    filename = function() {
      paste("best_models.csv")
    },
    content = function(file) {
      write.csv(best_models(), file, row.names = FALSE)
    }
  )
}

# Run app
shinyApp(ui = data_ui, server = data_server)

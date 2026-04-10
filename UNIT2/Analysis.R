# ============================================================
#  Heat Anomaly Detection Dashboard — R / Shiny
#  Equivalent of the Python Streamlit dashboard
#
#  Requirements (install once):
#    install.packages(c("shiny","shinydashboard","plotly",
#                       "dplyr","DT","fresh"))
#
#  Run with:
#    shiny::runApp("heat_detection_dashboard.R")
#  OR paste the whole file into the R console and call:
#    shinyApp(ui, server)
# ============================================================

library(shiny)
library(shinydashboard)
library(plotly)
library(dplyr)
library(DT)
library(fresh)   # custom CSS theming

# ── 0. Helper functions ────────────────────────────────────────────────────────

compute_confusion <- function(df, det_col) {
  label <- df[["label"]]
  pred  <- df[[det_col]]
  list(
    tp = sum(label == 1 & pred == 1),
    tn = sum(label == 0 & pred == 0),
    fp = sum(label == 0 & pred == 1),
    fn = sum(label == 1 & pred == 0)
  )
}

compute_metrics <- function(conf) {
  tp <- conf$tp; tn <- conf$tn; fp <- conf$fp; fn <- conf$fn
  total     <- tp + tn + fp + fn
  accuracy  <- if (total  > 0) (tp + tn) / total * 100 else 0
  precision <- if ((tp + fp) > 0) tp / (tp + fp) * 100  else 0
  recall    <- if ((tp + fn) > 0) tp / (tp + fn) * 100  else 0
  f1        <- if ((precision + recall) > 0)
    2 * precision * recall / (precision + recall) else 0
  list(
    accuracy  = round(accuracy,  2),
    precision = round(precision, 2),
    recall    = round(recall,    2),
    f1        = round(f1,        2)
  )
}

get_alert_level <- function(temp, threshold) {
  if      (temp < threshold)       list(label = "SAFE",    color = "#69db7c", css = "success")
  else if (temp < threshold + 5)   list(label = "LEVEL 1", color = "#ffd43b", css = "warning")
  else if (temp < threshold + 10)  list(label = "LEVEL 2", color = "#ff922b", css = "warning")
  else                             list(label = "LEVEL 3", color = "#ff6b6b", css = "danger")
}

accuracy_by_type <- function(df, det_col) {
  types <- c("Normal", "High Heat", "Pattern Spike")
  setNames(sapply(types, function(t) {
    sub <- df[df[["Actual_Data_Type"]] == t, ]
    if (nrow(sub) == 0) return(0)
    round(mean(sub[[det_col]] == sub[["label"]]) * 100, 2)
  }), types)
}

filter_data <- function(df, data_type) {
  if (data_type == "All") return(df)
  df[df[["Actual_Data_Type"]] == data_type, ]
}

# ── 1. Plotly chart builders ───────────────────────────────────────────────────

# Transparent background shared config
bg_trans  <- "rgba(0,0,0,0)"
grid_col  <- "rgba(255,255,255,0.08)"
font_white <- list(color = "white", size = 10)

temperature_gauge <- function(current_temp, threshold) {
  plot_ly(
    type  = "indicator",
    mode  = "gauge+number+delta",
    value = current_temp,
    delta = list(
      reference  = threshold,
      increasing = list(color = "#ff6b6b"),
      decreasing = list(color = "#69db7c")
    ),
    number = list(suffix = " °C", font = list(size = 44, color = "white")),
    gauge = list(
      axis = list(range = list(25, 55), tickwidth = 1,
                  tickcolor = "lightgray", tickfont = list(color = "white")),
      bar  = list(color = "#1e88e5"),
      bgcolor     = bg_trans,
      borderwidth = 0,
      steps = list(
        list(range = c(25,           threshold),      color = "#2e7d32"),
        list(range = c(threshold,    threshold + 5),  color = "#f9a825"),
        list(range = c(threshold+5,  threshold + 10), color = "#e65100"),
        list(range = c(threshold+10, 55),             color = "#b71c1c")
      ),
      threshold = list(
        line = list(color = "white", width = 3),
        thickness = 0.75,
        value = threshold
      )
    ),
    title = list(text = "Machine Temperature (°C)",
                 font = list(color = "white", size = 13))
  ) %>%
    layout(
      paper_bgcolor = bg_trans,
      plot_bgcolor  = bg_trans,
      font   = font_white,
      height = 270,
      margin = list(l=10, r=10, t=30, b=10)
    )
}

performance_gauge <- function(accuracy, model_label) {
  plot_ly(
    type  = "indicator",
    mode  = "gauge+number",
    value = accuracy,
    number = list(suffix = "%", font = list(size = 44, color = "white")),
    gauge = list(
      axis = list(range = list(0, 100), tickwidth = 1,
                  tickcolor = "lightgray", tickfont = list(color = "white")),
      bar  = list(color = "#43a047"),
      bgcolor = bg_trans, borderwidth = 0,
      steps = list(
        list(range = c(0,  60),  color = "#b71c1c"),
        list(range = c(60, 80),  color = "#e65100"),
        list(range = c(80, 90),  color = "#f9a825"),
        list(range = c(90, 100), color = "#2e7d32")
      )
    ),
    title = list(text = paste(model_label, "Accuracy (%)"),
                 font = list(color = "white", size = 13))
  ) %>%
    layout(
      paper_bgcolor = bg_trans,
      plot_bgcolor  = bg_trans,
      font = font_white, height = 270,
      margin = list(l=10, r=10, t=30, b=10)
    )
}

comparison_bar <- function(met_nn, met_lg) {
  metrics <- c("Accuracy", "Precision", "Recall", "F1-Score")
  nn_vals <- c(met_nn$accuracy, met_nn$precision, met_nn$recall, met_nn$f1)
  lg_vals <- c(met_lg$accuracy, met_lg$precision, met_lg$recall, met_lg$f1)
  
  plot_ly() %>%
    add_bars(x = metrics, y = nn_vals, name = "Neural Network",
             marker = list(color = "#1e88e5"),
             text = paste0(round(nn_vals,1), "%"), textposition = "outside") %>%
    add_bars(x = metrics, y = lg_vals, name = "Threshold Logic",
             marker = list(color = "#e53935"),
             text = paste0(round(lg_vals,1), "%"), textposition = "outside") %>%
    layout(
      barmode = "group",
      paper_bgcolor = bg_trans, plot_bgcolor = bg_trans,
      font = font_white, height = 270,
      margin = list(l=10, r=10, t=30, b=10),
      yaxis  = list(range = c(0, 115), ticksuffix = "%",
                    gridcolor = grid_col),
      legend = list(font = list(color = "white"), bgcolor = bg_trans)
    )
}

accuracy_by_type_bar <- function(nn_by_type, lg_by_type) {
  cats <- c("Normal", "High Heat", "Pattern Spike")
  plot_ly() %>%
    add_bars(x = cats, y = nn_by_type[cats],
             name = "Neural Network",   marker = list(color = "#1e88e5")) %>%
    add_bars(x = cats, y = lg_by_type[cats],
             name = "Threshold Logic",  marker = list(color = "#e53935")) %>%
    layout(
      barmode = "group",
      paper_bgcolor = bg_trans, plot_bgcolor = bg_trans,
      font = font_white, height = 300,
      margin = list(l=10, r=10, t=40, b=10),
      title  = list(text = "Detection Accuracy by Data Type",
                    font = list(color = "white")),
      yaxis  = list(range = c(0, 115), ticksuffix = "%",
                    gridcolor = grid_col),
      legend = list(font = list(color = "white"), bgcolor = bg_trans)
    )
}

temperature_boxplot <- function(df, threshold) {
  types  <- c("Normal", "Pattern Spike", "High Heat")
  colors <- c("Normal" = "#92C5DE", "Pattern Spike" = "#F4A582",
              "High Heat" = "#D6604D")
  p <- plot_ly(height = 300)
  for (t in types) {
    sub <- df[df[["Actual_Data_Type"]] == t, "max_temp_recorded"]
    p <- add_trace(p, type = "box", y = sub, name = t,
                   marker = list(color = colors[[t]]),
                   line   = list(color = colors[[t]]),
                   boxmean = TRUE)
  }
  p %>%
    add_segments(x = 0, xend = 1, y = threshold, yend = threshold,
                 line = list(color = "#ffd43b", dash = "dash"),
                 name = paste0("Threshold (", threshold, "°C)"),
                 inherit = FALSE) %>%
    layout(
      paper_bgcolor = bg_trans, plot_bgcolor = bg_trans,
      font = font_white,
      margin = list(l=10, r=10, t=40, b=10),
      title = list(text = "Temperature Distribution by Category",
                   font = list(color = "white")),
      yaxis = list(title = "Max Temp (°C)", gridcolor = grid_col),
      legend = list(font = list(color = "white"), bgcolor = bg_trans)
    )
}

outcome_scatter <- function(df, perf_col, title_text) {
  outcomes <- list(
    "Correct (Detected)"         = list(color = "#1e88e5", symbol = "circle"),
    "Correct (Normal)"           = list(color = "#43a047", symbol = "square"),
    "Missed (Pattern Blind)"     = list(color = "#e53935", symbol = "x"),
    "Missed (Threshold Failure)" = list(color = "#ff922b", symbol = "diamond"),
    "False Alarm"                = list(color = "#ab47bc", symbol = "triangle-up")
  )
  p <- plot_ly(height = 300)
  for (outcome in names(outcomes)) {
    sub <- df[df[[perf_col]] == outcome, ]
    if (nrow(sub) == 0) next
    p <- add_trace(p, type = "scatter", mode = "markers",
                   x = as.integer(rownames(sub)),
                   y = sub[["max_temp_recorded"]],
                   name = outcome,
                   marker = list(
                     color  = outcomes[[outcome]]$color,
                     symbol = outcomes[[outcome]]$symbol,
                     size   = 6, opacity = 0.8,
                     line   = list(width = 0.4, color = "black")
                   ))
  }
  p %>%
    add_segments(x = min(as.integer(rownames(df))),
                 xend = max(as.integer(rownames(df))),
                 y = 38, yend = 38,
                 line = list(color = "white", dash = "dash"),
                 opacity = 0.4, name = "~38°C threshold",
                 inherit = FALSE) %>%
    layout(
      paper_bgcolor = bg_trans, plot_bgcolor = bg_trans,
      font = font_white,
      margin = list(l=10, r=10, t=40, b=10),
      title  = list(text = title_text, font = list(color = "white")),
      xaxis  = list(title = "Sample Index", gridcolor = grid_col),
      yaxis  = list(title = "Max Temp (°C)", gridcolor = grid_col),
      legend = list(font = list(color = "white", size = 9),
                    bgcolor = bg_trans)
    )
}

radar_chart <- function(met_nn, met_lg) {
  cats   <- c("Accuracy", "Precision", "Recall", "F1-Score")
  nn_raw <- c(met_nn$accuracy, met_nn$precision, met_nn$recall, met_nn$f1)
  lg_raw <- c(met_lg$accuracy, met_lg$precision, met_lg$recall, met_lg$f1)
  nn_norm <- nn_raw / 100
  lg_norm <- lg_raw / 100
  
  plot_ly(height = 340) %>%
    add_trace(type = "scatterpolar",
              r = c(nn_norm, nn_norm[1]), theta = c(cats, cats[1]),
              fill = "toself", fillcolor = "#1e88e5", opacity = 0.25,
              name = "Neural Network",
              line = list(color = "#1e88e5", width = 2)) %>%
    add_trace(type = "scatterpolar",
              r = c(lg_norm, lg_norm[1]), theta = c(cats, cats[1]),
              fill = "toself", fillcolor = "#e53935", opacity = 0.25,
              name = "Threshold Logic",
              line = list(color = "#e53935", width = 2)) %>%
    layout(
      paper_bgcolor = bg_trans, plot_bgcolor = bg_trans,
      font = font_white,
      margin = list(l=10, r=10, t=40, b=10),
      title  = list(text = "Precision / Recall / F1 / Accuracy",
                    font = list(color = "white")),
      polar  = list(
        bgcolor = bg_trans,
        radialaxis  = list(visible = TRUE, range = c(0,1),
                           tickfont = list(color = "grey", size = 8),
                           gridcolor = grid_col),
        angularaxis = list(tickfont = list(color = "white", size = 10),
                           gridcolor = grid_col)
      ),
      legend = list(font = list(color = "white"), bgcolor = bg_trans)
    )
}

# ── 2. UI ─────────────────────────────────────────────────────────────────────

# Dark theme via fresh
dark_theme <- create_theme(
  adminlte_color(
    light_blue = "#1e88e5"
  ),
  adminlte_sidebar(
    dark_bg     = "#1a1d27",
    dark_color  = "#c9d1d9",
    dark_hover_bg    = "#2e3250",
    dark_hover_color = "#ffffff"
  ),
  adminlte_global(
    content_bg  = "#0f1117",
    box_bg      = "#1e2130",
    info_box_bg = "#1e2130"
  )
)

panel_header_css <- "
  .panel-hdr {
    border-radius: 8px 8px 0 0;
    padding: 8px 14px;
    font-weight: 700;
    font-size: 0.95rem;
    color: white;
    margin-bottom: 0;
  }
  .ph-blue   { background-color: #1565c0; }
  .ph-green  { background-color: #2e7d32; }
  .ph-orange { background-color: #e65100; }
  .ph-purple { background-color: #6a0dad; }
  .ph-teal   { background-color: #00695c; }
  .ph-red    { background-color: #b71c1c; }
  .panel-body-dark {
    background-color: #1e2130;
    border-radius: 0 0 8px 8px;
    padding: 14px;
    margin-bottom: 16px;
  }
  .reading-row {
    display: flex;
    justify-content: space-between;
    padding: 5px 0;
    border-bottom: 1px solid #2e3250;
    font-size: 0.88rem;
  }
  .reading-label { color: #8b949e; }
  .reading-value { color: #ffffff; font-weight: 600; }
  .badge-active {
    background-color:#43a047; color:white; padding:4px 12px;
    border-radius:20px; font-size:0.75rem; font-weight:700;
  }
  body, .content-wrapper { background-color: #0f1117 !important; }
  .box { background: #1e2130 !important; border-top: none !important; }
  .box-header { background: #1e2130 !important; color: white !important; }
  .sidebar-menu > li > a { color: #c9d1d9 !important; }
"

ui <- dashboardPage(
  skin = "black",
  
  dashboardHeader(
    title = tags$span(
      style = "font-weight:700;",
      "⚙️ Heat Anomaly Detection"
    )
  ),
  
  dashboardSidebar(
    use_theme(dark_theme),
    tags$head(tags$style(HTML(panel_header_css))),
    
    sidebarMenu(
      menuItem("📊 Dashboard Overview",   tabName = "overview", selected = TRUE),
      menuItem("🔬 Sensor Monitor",        tabName = "sensor"),
      menuItem("📈 Detailed Analysis",     tabName = "analysis"),
      menuItem("⚙️ Model Comparison",      tabName = "model")
    ),
    
    hr(),
    tags$div(
      style = "padding: 0 15px;",
      tags$b(style = "color:#c9d1d9;", "Filter by Data Type"),
      selectInput("data_filter", label = NULL,
                  choices = c("All","Normal","High Heat","Pattern Spike"),
                  selected = "All"),
      
      tags$b(style = "color:#c9d1d9;", "Alert Threshold (°C)"),
      sliderInput("threshold", label = NULL,
                  min = 30, max = 50, value = 38, step = 1),
      
      tags$b(style = "color:#c9d1d9;", "Active Model"),
      radioButtons("model_choice", label = NULL,
                   choices = c("Neural Network","Threshold Logic"),
                   selected = "Neural Network"),
      
      checkboxInput("auto_refresh", "Auto Refresh (5s)", value = FALSE),
      hr(),
      actionButton("refresh_btn", "🔄 Refresh Data",
                   class = "btn-block btn-default"),
      hr(),
      uiOutput("last_updated")
    )
  ),
  
  dashboardBody(
    # ── KPI Row ──
    fluidRow(
      valueBoxOutput("kpi_temp",   width = 3),
      valueBoxOutput("kpi_alert",  width = 3),
      valueBoxOutput("kpi_acc",    width = 3),
      valueBoxOutput("kpi_alerts", width = 3)
    ),
    
    # ── Gauges Row ──
    fluidRow(
      column(6,
             tags$div(class = "panel-hdr ph-blue",
                      "🌡️ Temperature Gauge — Last Reading"),
             tags$div(class = "panel-body-dark",
                      plotlyOutput("temp_gauge", height = "270px"),
                      uiOutput("temp_readings")
             )
      ),
      column(6,
             tags$div(class = "panel-hdr ph-green",
                      "🤖 Model Performance Gauge"),
             tags$div(class = "panel-body-dark",
                      plotlyOutput("perf_gauge", height = "270px"),
                      uiOutput("perf_readings")
             )
      )
    ),
    
    # ── Live Table + Comparison Bar ──
    fluidRow(
      column(6,
             tags$div(class = "panel-hdr ph-orange",
                      "📋 Live Sensor Readings — Last 10 Samples"),
             tags$div(class = "panel-body-dark",
                      DTOutput("live_table")
             )
      ),
      column(6,
             tags$div(class = "panel-hdr ph-purple",
                      "📊 NN vs Logic — Metric Comparison"),
             tags$div(class = "panel-body-dark",
                      plotlyOutput("comparison_bar_plot", height = "300px")
             )
      )
    ),
    
    # ── Accuracy by Type + Box Plot ──
    fluidRow(
      column(6,
             tags$div(class = "panel-hdr ph-teal",
                      "🎯 Detection Accuracy by Data Type"),
             tags$div(class = "panel-body-dark",
                      plotlyOutput("acc_type_bar", height = "300px")
             )
      ),
      column(6,
             tags$div(class = "panel-hdr ph-red",
                      "📦 Temperature Distribution by Category"),
             tags$div(class = "panel-body-dark",
                      plotlyOutput("temp_boxplot", height = "300px")
             )
      )
    ),
    
    # ── Scatter Plots ──
    tags$div(class = "panel-hdr ph-blue",
             "🔍 Detection Outcomes — Max Temp vs Sample Index"),
    tags$div(class = "panel-body-dark",
             fluidRow(
               column(6, plotlyOutput("scatter_nn",    height = "300px")),
               column(6, plotlyOutput("scatter_logic", height = "300px"))
             )
    ),
    tags$br(),
    
    # ── Radar Chart ──
    tags$div(class = "panel-hdr ph-purple",
             "🕸️ Radar Chart — All Metrics Overview"),
    tags$div(class = "panel-body-dark",
             fluidRow(
               column(4),
               column(4, plotlyOutput("radar", height = "340px")),
               column(4)
             )
    ),
    tags$br()
  )
)

# ── 3. Server ──────────────────────────────────────────────────────────────────

server <- function(input, output, session) {
  
  # ── Load data (reactive to refresh button) ──
  df_full <- reactiveVal(NULL)
  
  observe({
    input$refresh_btn   # dependency
    # Put the CSV next to this script, or adjust the path below
    path <- "data/model_vs_logic_comparison.csv"
    if (file.exists(path)) {
      df_full(read.csv(path, stringsAsFactors = FALSE))
    } else {
      showNotification(
        paste("CSV not found at:", normalizePath(path, mustWork = FALSE),
              "— place model_vs_logic_comparison.csv in a 'data/' sub-folder."),
        type = "error", duration = NULL
      )
    }
  })
  
  # ── Auto-refresh every 5 s ──
  observe({
    if (isTRUE(input$auto_refresh)) {
      invalidateLater(5000, session)
      input$refresh_btn   # trigger the same reload
    }
  })
  
  # ── Filtered data ──
  df <- reactive({
    req(df_full())
    filter_data(df_full(), input$data_filter)
  })
  
  # ── Derived computations ──
  conf_nn <- reactive({ compute_confusion(df(), "nn_detected")    })
  conf_lg <- reactive({ compute_confusion(df(), "logic_detected") })
  met_nn  <- reactive({ compute_metrics(conf_nn()) })
  met_lg  <- reactive({ compute_metrics(conf_lg()) })
  met_active <- reactive({
    if (input$model_choice == "Neural Network") met_nn() else met_lg()
  })
  
  current_temp <- reactive({ tail(df()[["max_temp_recorded"]], 1) })
  avg_temp     <- reactive({ mean(df()[["max_temp_recorded"]]) })
  total_alerts <- reactive({ sum(df()[["label"]]) })
  alert        <- reactive({ get_alert_level(current_temp(), input$threshold) })
  
  nn_by_type <- reactive({ accuracy_by_type(df(), "nn_detected")    })
  lg_by_type <- reactive({ accuracy_by_type(df(), "logic_detected") })
  
  # ── KPI boxes ──
  output$kpi_temp <- renderValueBox({
    valueBox(
      value    = paste0(round(current_temp(), 1), " °C"),
      subtitle = "📌 Current Max Temperature",
      icon     = icon("thermometer-half"),
      color    = "orange"
    )
  })
  
  output$kpi_alert <- renderValueBox({
    al <- alert()
    col_map <- c(SAFE = "green", "LEVEL 1" = "yellow",
                 "LEVEL 2" = "orange", "LEVEL 3" = "red")
    valueBox(
      value    = al$label,
      subtitle = "🔔 Current Alert Level",
      icon     = icon("bell"),
      color    = col_map[[al$label]]
    )
  })
  
  output$kpi_acc <- renderValueBox({
    valueBox(
      value    = paste0(met_active()$accuracy, "%"),
      subtitle = paste("🤖", input$model_choice, "Accuracy"),
      icon     = icon("robot"),
      color    = "blue"
    )
  })
  
  output$kpi_alerts <- renderValueBox({
    valueBox(
      value    = total_alerts(),
      subtitle = "⚠️ Total Alerts in Dataset",
      icon     = icon("exclamation-triangle"),
      color    = "purple"
    )
  })
  
  # ── Gauges ──
  output$temp_gauge <- renderPlotly({
    temperature_gauge(current_temp(), input$threshold)
  })
  
  output$perf_gauge <- renderPlotly({
    performance_gauge(met_active()$accuracy, input$model_choice)
  })
  
  # ── Reading rows below gauges ──
  reading_row <- function(label, value) {
    tags$div(class = "reading-row",
             tags$span(class = "reading-label", label),
             tags$span(class = "reading-value", value)
    )
  }
  
  output$temp_readings <- renderUI({
    al <- alert()
    tagList(
      reading_row("Latest Reading",  paste0(round(current_temp(), 2), " °C")),
      reading_row("Dataset Average", paste0(round(avg_temp(), 2),    " °C")),
      reading_row("Set Threshold",   paste0(input$threshold,          " °C")),
      tags$div(class = "reading-row",
               tags$span(class = "reading-label", "Alert Status"),
               tags$span(style = paste0("color:", al$color, "; font-weight:700;"),
                         al$label)
      )
    )
  })
  
  output$perf_readings <- renderUI({
    m <- met_active()
    tagList(
      reading_row("Accuracy",  paste0(m$accuracy,  "%")),
      reading_row("Precision", paste0(m$precision, "%")),
      reading_row("Recall",    paste0(m$recall,    "%")),
      reading_row("F1-Score",  paste0(m$f1,        "%"))
    )
  })
  
  # ── Live sensor table ──
  output$live_table <- renderDT({
    cols_sel <- c("t1","t2","t3","t4","t5","max_temp_recorded","label","Actual_Data_Type")
    last10 <- tail(df()[, cols_sel], 10)
    colnames(last10) <- c("T1","T2","T3","T4","T5","Max Temp","Alert","Type")
    last10$Alert <- ifelse(last10$Alert == 1, "⚠️ YES", "✅ NO")
    
    datatable(last10, rownames = FALSE,
              options = list(
                dom        = "t",
                pageLength = 10,
                scrollX    = TRUE,
                initComplete = JS(
                  "function(settings, json) {",
                  "  $(this.api().table().header()).css({'background-color':'#1e2130','color':'white'});",
                  "}"
                )
              )) %>%
      formatStyle(columns = colnames(last10),
                  color = "white", backgroundColor = "#1e2130")
  }, server = FALSE)
  
  # ── Comparison bar ──
  output$comparison_bar_plot <- renderPlotly({
    comparison_bar(met_nn(), met_lg())
  })
  
  # ── Accuracy by type bar ──
  output$acc_type_bar <- renderPlotly({
    accuracy_by_type_bar(nn_by_type(), lg_by_type())
  })
  
  # ── Box plot ──
  output$temp_boxplot <- renderPlotly({
    temperature_boxplot(df(), input$threshold)
  })
  
  # ── Scatter plots ──
  output$scatter_nn <- renderPlotly({
    outcome_scatter(df(), "NN_Performance", "Neural Network (Proposed)")
  })
  
  output$scatter_logic <- renderPlotly({
    outcome_scatter(df(), "Paper_Logic_Performance", "Threshold Logic (Base Paper)")
  })
  
  # ── Radar chart ──
  output$radar <- renderPlotly({
    radar_chart(met_nn(), met_lg())
  })
  
  # ── Sidebar footer ──
  output$last_updated <- renderUI({
    tags$small(style = "color:#555;",
               paste("Updated:", format(Sys.time(), "%H:%M:%S")), tags$br(),
               "Source: model_vs_logic_comparison.csv"
    )
  })
}

# ── 4. Launch ─────────────────────────────────────────────────────────────────
shinyApp(ui, server)
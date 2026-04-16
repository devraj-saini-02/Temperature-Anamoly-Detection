# Load the required libraries
library(shiny)
library(shinydashboard)
library(dplyr)
library(DT)

# --- BACKEND LOGIC (Data Processing) ---
# Read the CSV file (Ensure the CSV is in the same folder as this script)
df <- read.csv("model_vs_logic_comparison.csv", stringsAsFactors = FALSE)

# Recreate your PySpark Logic in R
logic_victory_df <- df %>% 
  filter(label == 1 & logic_detected == 1 & nn_detected == 0)

nn_victory_df <- df %>% 
  filter(label == 1 & nn_detected == 1 & logic_detected == 0)

summary_df <- df %>%
  group_by(Actual_Data_Type) %>%
  summarize(
    NN_Success_Rate = mean(nn_detected, na.rm = TRUE),
    Logic_Success_Rate = mean(logic_detected, na.rm = TRUE)
  )
# Note: I omitted the "remove last row" logic as it usually depends on random partition order in Spark. 
# If you explicitly want to drop the last row in R, you would add: %>% slice(1:(n()-1))

# --- UI (User Interface) ---
# --- UI (User Interface) ---
# --- UI (User Interface) ---
ui <- dashboardPage(
  dashboardHeader(title = "Anomaly Integrity Audit"),
  dashboardSidebar(disable = TRUE), 
  dashboardBody(
    # Top Row: KPI Boxes
    fluidRow(
      valueBox(
        nrow(logic_victory_df), 
        "High-Criticality Misses by NN (Logic Saved the Day)", 
        icon = icon("exclamation-triangle"), color = "red", width = 6
      ),
      valueBox(
        nrow(nn_victory_df), 
        "Pattern Misses by Logic (NN Saved the Day)", 
        icon = icon("brain"), color = "purple", width = 6
      )
    ),
    
    # Second Row: The two detailed tables
    fluidRow(
      box(
        title = "Logic Superiority (Deterministic Thresholds)", 
        status = "danger", solidHeader = TRUE, width = 6,
        DTOutput("logicTable")
      ),
      box(
        title = "NN Superiority (Subtle Pattern Detection)", 
        status = "primary", solidHeader = TRUE, width = 6,
        DTOutput("nnTable")
      )
    ),
    
    # Third Row: Summary Table
    fluidRow(
      box(
        title = "Average Detection Rates by Data Type", 
        status = "success", solidHeader = TRUE, width = 12,
        DTOutput("summaryTable")
      )
    ),
    
    # FOURTH ROW: First 3 Pictures (No Titles)
    fluidRow(
      box(width = 4, tags$img(src = "pic1.png", style = "width: 100%;")),
      box(width = 4, tags$img(src = "pic2.png", style = "width: 100%;")),
      box(width = 4, tags$img(src = "pic3.png", style = "width: 100%;"))
    ),
    
    # FIFTH ROW: Last 3 Pictures (No Titles)
    fluidRow(
      box(width = 4, tags$img(src = "pic4.png", style = "width: 100%;")),
      box(width = 4, tags$img(src = "pic5.png", style = "width: 100%;")),
      box(width = 4, tags$img(src = "pic6.png", style = "width: 100%;"))
    )
  )
)

# --- SERVER (How the UI gets its data) ---
server <- function(input, output) {
  
  # Render the Logic Victory Table (showing specific columns and top 6 rows)
  output$logicTable <- renderDT({
    logic_victory_df %>% 
      select(max_temp_recorded, Actual_Data_Type) %>% 
      head(6) %>%
      datatable(options = list(dom = 't')) # 't' hides the search bar for a cleaner look
  })
  
  # Render the NN Victory Table (showing specific columns and top 5 rows)
  output$nnTable <- renderDT({
    nn_victory_df %>% 
      select(t1, t2, t3, t4, t5, Actual_Data_Type) %>% 
      head(5) %>%
      datatable(options = list(dom = 't'))
  })
  
  # Render the Distributed Aggregation Summary
  output$summaryTable <- renderDT({
    summary_df %>%
      datatable(options = list(pageLength = 5))
  })
}

# Run the application 
shinyApp(ui = ui, server = server)
# Use an official R Shiny image as a parent image
FROM rocker/shiny:latest

# Install Linux system dependencies required by R packages
RUN apt-get update && apt-get install -y \
    libcurl4-gnutls-dev \
    libcairo2-dev \
    libxt-dev \
    libssl-dev \
    libssh2-1-dev

# Install the necessary R packages
RUN R -e "install.packages(c('shiny', 'shinydashboard', 'dplyr', 'DT'), repos='https://cloud.r-project.org/')"

# Remove the default Shiny apps that come with the base image
RUN rm -rf /srv/shiny-server/*

# Copy your app.R and the CSV file into the Docker container
COPY app.R /srv/shiny-server/
COPY model_vs_logic_comparison.csv /srv/shiny-server/

# Make the Shiny server directory readable
RUN chown -R shiny:shiny /srv/shiny-server

# Expose the port Shiny runs on
EXPOSE 3838

# Run the Shiny server
CMD ["/usr/bin/shiny-server"]
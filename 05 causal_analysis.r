# Install required packages
install.packages("plm")
install.packages("dplyr")
install.packages("readr")
install.packages("rstudioapi")
install.packages("Hmisc")
install.packages("car")
install.packages("lmtest")


# Load libraries
library(plm)
library(dplyr)
library(readr)
library(rstudioapi)
library(Hmisc)
library(car)
library(lmtest)
library(sandwich)
library(ggplot2)


# Set current working directory
if (rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
} else {
  stop("Not running in RStudio. Please set the working directory manually.")
}

# Load the dataset
data_file <- "data/FINAL_agg_Dataset.csv"  # This should be the final aggregated dataset created after running 04 street_data_exploration.ipynb
if (file.exists(data_file)) {
  agg_data <- read_csv(data_file)
} else {
  stop(paste("File not found:", data_file))
}


# ------------------------------------------------------------------------------

# TRUST CAUSAL ANALYSIS

# DATA PROCESSING ----------------------------------------------

# Create a time variable based on year and quarter, arrange data, convert to pdata.frame, and remove year and quarter columns
data_trust <- agg_data %>%
  mutate(time = as.Date(paste0(year, "-", (quarter - 1) * 3 + 1, "-01"), format = "%Y-%m-%d")) %>%
  arrange(borough, time) %>%
  pdata.frame(index = c("borough", "time")) %>%
  subset(select = -c(year, quarter, GoodJoblocal, total_q_crimes))

# Convert to data frame
df <- as.data.frame(data_trust)

# Data descriptors
describe(df)
summary(df)


# MODELS -------------------------------------------------------

# Create the model formula for trust
factors_t <- c(colnames(df[, -which(names(df) %in% c("TrustMPS"))]))
eq_t <- as.formula(paste("TrustMPS ~ ", paste(factors_t, collapse="+")))


# Fit a linear model to check for multicollinearity
lm_model <- lm(eq_t, data = df)

# Identify and list aliased (perfectly collinear) coefficients
aliased <- alias(lm_model)$Complete

if (length(aliased) > 0) {
  aliased_vars <- names(which(apply(aliased, 2, function(x) any(x != 0))))
  cat(paste("Aliased variables detected: \n", paste(aliased_vars, collapse="\n"), "\n"))

  # Remove aliased variables from the data frame
  df <- df %>% select(-all_of(aliased_vars))

  # Update the equation explicitly by removing collinear features
  eq_t <- as.formula(TrustMPS ~ borough
                     + time
                     + Contactwardofficer
                     + Informedlocal
                     + Listentoconcerns
                     + Reliedontobethere
                     + Treateveryonefairly
                     + Understandissues
                     + Gender_Female
                     + Gender_Male
                     + Gender_Other
                     + Agerange_18.24
                     + Agerange_25.34
                     + Agerange_over34
                     + Objectofsearch_Articlesforuseincriminaldamage
                     + Objectofsearch_EvidenceofoffencesundertheAct
                     + Objectofsearch_Firearms
                     + Objectofsearch_Offensiveweapons
                     + Objectofsearch_Stolengoods
                     + violent_q_crimes
                     + property_q_crimes
                     + public_order_q_crimes
                     + drug_or_weapon_q_crimes
                     + other_q_crimes)

  # Re-fit the linear model
  lm_model <- lm(eq_t, data = df)

} else {
  print("No aliased variables detected.")
}

# Calculate VIF values to check for multicollinearity
vif_values <- vif(lm_model)
print(names(vif_values))


# Fit the Pooled OLS model
pool_model <- plm(eq_t, index = c("borough", "time"), data = df, model = "pooling")

# Fit models with time effects, entity effects, and mixed effects to check for time/entity effect
pool_entity_eff <- plm(update(eq_t, reformulate(c(".", "borough"))), index = c("borough", "time"), data = df, model = "pooling")
pool_mixed_eff <- plm(update(eq_t, reformulate(c(".", "borough + time"))), index = c("borough", "time"), data = df, model = "pooling")
pool_time_eff <- plm(update(eq_t, reformulate(c(".", "time"))), index = c("borough", "time"), data = df, model = "pooling")

# Test for fixed effects using pFtest
sprintf("Fixed entity effect in data: %s", pFtest(pool_entity_eff, pool_model)$p.value <= 0.05)
sprintf("Fixed mixed effect in data: %s", pFtest(pool_mixed_eff, pool_model)$p.value <= 0.05)
sprintf("Fixed time effect in data: %s", pFtest(pool_time_eff, pool_model)$p.value <= 0.05)
# TRUE would suggest there are significant differences in the intercepts across different entities and/or time that are not captured by the pooled OLS model
# RESULT: First two tests return TRUE which indicates Entity effects and Time effects are not accounted for in the OLS model

# Test for random effects (Breush-Pagan 1980)
sprintf("Random entity effect in data: %s", plmtest(pool_model, type="bp", effect="individual")$p.value <= 0.05)
sprintf("Random time effect in data: %s", plmtest(pool_model, type="bp", effect="time")$p.value<= 0.05)
sprintf("Random mixed effect in data: %s", plmtest(pool_model, type="ghm", effect="twoways")$p.value<= 0.05) # ghm is only available for twoways, but robust for unbalanced panel

# Fit fixed effects and random effects model
fe_model <- plm(eq_t, index = c("borough", "time"), data = df, model = "within")
summary(fe_model)

re_model <- plm(eq_t, index = c("borough", "time"), data = df, model = "random", random.method = "walhus")
summary(re_model)


# Conduct Hausman test to choose between FE and RE models
h_test <- phtest(fe_model, re_model)
summary(h_test)
p_value <- h_test$p.value
if(p_value < 0.05) {
  print("p-value is less than 0.05, so we reject the null hypothesis. Fixed effects (FE) model is preferred.")
  chosen_model = fe_model
} else {
  print("p-value is greater than 0.05, so we do not reject the null hypothesis. Random effects (RE) model is preferred.")
  chosen_model = re_model
}


# Test for autocorrelation
auto_corr_test <- pbgtest(chosen_model)
p_value <- auto_corr_test$p.value
if(p_value < 0.05) {
  print("p-value is less than 0.05, so we reject the null hypothesis. There is serial correlation. ")
} else {
  print("p-value is greater than 0.05, so we do not reject the null hypothesis. There is autocorrelation. ")
}


# Test for heteroskedasticity
het_test <- bptest(eq_t, data=df, studentize=F)
p_value <- het_test$p.value
if(p_value < 0.05) {
  print("p-value is less than 0.05, so we reject the null hypothesis. There is heteroskedasticity. ")
} else {
  print("p-value is greater than 0.05, so we do not reject the null hypothesis. There is homoskedasticity. ")
}

# Calculate robust standard errors
coeftest(fe_model, vcovHC(re_model, method= "arellano"))


# BAR CHART TO DISPLAY COEFFICIENTS ---------------------------

# Variables of interest
variables_of_interest <- c("Contactwardofficer", "Listentoconcerns",
                           "Reliedontobethere", "Treateveryonefairly", "Understandissues")

# Extract coefficients from model summary for variables of interest
coefficients_summary <- coef(summary(re_model))
coefficients_of_interest <- coefficients_summary[rownames(coefficients_summary) %in% variables_of_interest, "Estimate"]
p_values_of_interest <- coefficients_summary[rownames(coefficients_summary) %in% variables_of_interest, "Pr(>|z|)"]

# Create a data frame for plotting
results_df <- data.frame(variable = names(coefficients_of_interest),
                         coefficient = coefficients_of_interest,
                         p_value = p_values_of_interest,
                         stringsAsFactors = FALSE)

# Sort by coefficient value for better visualization
results_df <- results_df[order(abs(results_df$coefficient), decreasing = TRUE),]

# Plot
ggplot(results_df, aes(x = reorder(variable, coefficient), y = coefficient, fill = p_value < 0.001)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = sprintf("%.7f", coefficient)), vjust = -0.5, size = 3) +
  coord_flip() +
  labs(x = "Variables", y = "Coefficient Estimate",
       title = "Top 5 significant features (p < 0.1)",
       subtitle = "High statistical significance (p < 0.001) indicated by color") +
  scale_fill_manual(values = c("TRUE" = "lightblue", "FALSE" = "gray")) +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 10))





# ------------------------------------------------------------------------------

# CONFIDENCE CAUSAL ANALYSIS

# DATA PROCESSING ----------------------------------------------

# Create a time variable based on year and quarter, arrange data, convert to pdata.frame, and remove year and quarter columns
data_confidence <- agg_data %>%
  mutate(time = as.Date(paste0(year, "-", (quarter - 1) * 3 + 1, "-01"), format = "%Y-%m-%d")) %>%
  arrange(borough, time) %>%
  pdata.frame(index = c("borough", "time")) %>%
  subset(select = -c(year, quarter, TrustMPS, total_q_crimes))

# Convert to data frame
df <- as.data.frame(data_confidence, index = c("borough", "time"))

# Data descriptors
describe(df)
summary(df)


# MODELS -------------------------------------------------------

# Create the model formula for confidence
factors_c <- c(colnames(df[, -which(names(df) %in% c("GoodJoblocal"))]))
eq_c <- as.formula(paste("GoodJoblocal ~ ", paste(factors_c, collapse="+")))

# Fit a linear model to check for multicollinearity
lm_model <- lm(eq_c, data = df)

# Identify and list aliased (perfectly collinear) coefficients
aliased <- alias(lm_model)$Complete

if (length(aliased) > 0) {
  aliased_vars <- names(which(apply(aliased, 2, function(x) any(x != 0))))
  cat(paste("Aliased variables detected: \n", paste(aliased_vars, collapse="\n"), "\n"))

  # Remove aliased variables from the data frame
  df <- df %>% select(-all_of(aliased_vars))

  # Update the equation explicitly by removing collinear features
  eq_c <- as.formula(GoodJoblocal ~ borough
                     + time
                     + Contactwardofficer
                     + Informedlocal
                     + Listentoconcerns
                     + Reliedontobethere
                     + Treateveryonefairly
                     + Understandissues
                     + Gender_Female
                     + Gender_Male
                     + Gender_Other
                     + Agerange_18.24
                     + Agerange_25.34
                     + Agerange_over34
                     + Objectofsearch_Articlesforuseincriminaldamage
                     + Objectofsearch_EvidenceofoffencesundertheAct
                     + Objectofsearch_Firearms
                     + Objectofsearch_Offensiveweapons
                     + Objectofsearch_Stolengoods
                     + violent_q_crimes
                     + property_q_crimes
                     + public_order_q_crimes
                     + drug_or_weapon_q_crimes
                     + other_q_crimes)

  # Re-fit the linear model
  lm_model <- lm(eq_c, data = df)

} else {
  print("No aliased variables detected.")
}

# Calculate VIF values to check for multicollinearity
vif_values <- vif(lm_model)
print(names(vif_values))

# Fit the Pooled OLS model
pool_model <- plm(eq_c, index = c("borough", "time"), data = df, model = "pooling")

# Fit models with time effects, entity effects, and mixed effects to check for time/entity effect
pool_entity_eff <- plm(update(eq_c, reformulate(c(".", "borough"))), index = c("borough", "time"), data = df, model = "pooling")
pool_mixed_eff <- plm(update(eq_c, reformulate(c(".", "borough + time"))), index = c("borough", "time"), data = df, model = "pooling")
pool_time_eff <- plm(update(eq_c, reformulate(c(".", "time"))), index = c("borough", "time"), data = df, model = "pooling")

# Test for fixed effects using pFtest
sprintf("Fixed entity effect in data: %s", pFtest(pool_entity_eff, pool_model)$p.value <= 0.05)
sprintf("Fixed mixed effect in data: %s", pFtest(pool_mixed_eff, pool_model)$p.value <= 0.05)
sprintf("Fixed time effect in data: %s", pFtest(pool_time_eff, pool_model)$p.value <= 0.05)
# TRUE would suggest there are significant differences in the intercepts across different entities and/or time that are not captured by the pooled OLS model
# RESULT: First two tests return TRUE which indicates Entity effects and Time effects are not accounted for in the OLS model

# Test for random effects (Breusch-Pagan 1980)
sprintf("Random entity effect in data: %s", plmtest(pool_model, type="bp", effect="individual")$p.value <= 0.05)
sprintf("Random time effect in data: %s", plmtest(pool_model, type="bp", effect="time")$p.value<= 0.05)
sprintf("Random mixed effect in data: %s", plmtest(pool_model, type="ghm", effect="twoways")$p.value<= 0.05) # ghm is only available for twoways, but robust for unbalanced panel

# Fit fixed effects and random effects model
fe_model <- plm(eq_c, index = c("borough", "time"), data = df, model = "within")
summary(fe_model)

re_model <- plm(eq_c, index = c("borough", "time"), data = df, model = "random", random.method = "walhus")
summary(re_model)

# Conduct Hausman test to choose between FE and RE models
h_test <- phtest(fe_model, re_model)
summary(h_test)
p_value <- h_test$p.value
if(p_value < 0.05) {
  print("p-value is less than 0.05, so we reject the null hypothesis. Fixed effects (FE) model is preferred.")
  chosen_model = fe_model
} else {
  print("p-value is greater than 0.05, so we do not reject the null hypothesis. Random effects (RE) model is preferred.")
  chosen_model = re_model
}

# Test for autocorrelation
auto_corr_test <- pbgtest(chosen_model)
p_value <- auto_corr_test$p.value
if(p_value < 0.05) {
  print("p-value is less than 0.05, so we reject the null hypothesis. There is serial correlation. ")
} else {
  print("p-value is greater than 0.05, so we do not reject the null hypothesis. There is autocorrelation. ")
}

# Test for heteroskedasticity
het_test <- bptest(eq_c, data=df, studentize=F)
p_value <- het_test$p.value
if(p_value < 0.05) {
  print("p-value is less than 0.05, so we reject the null hypothesis. There is heteroskedasticity. ")
} else {
  print("p-value is greater than 0.05, so we do not reject the null hypothesis. There is homoskedasticity. ")
}

# Calculate robust standard errors
coeftest(fe_model, vcovHC(re_model, method= "arellano"))

# BAR CHART TO DISPLAY COEFFICIENTS ---------------------------

# Variables of interest
variables_of_interest <- c("Contactwardofficer", "Listentoconcerns",
                           "Reliedontobethere", "Treateveryonefairly", "Understandissues")

# Extract coefficients from model summary for variables of interest
coefficients_summary <- coef(summary(re_model))
coefficients_of_interest <- coefficients_summary[rownames(coefficients_summary) %in% variables_of_interest, "Estimate"]
p_values_of_interest <- coefficients_summary[rownames(coefficients_summary) %in% variables_of_interest, "Pr(>|z|)"]

# Create a data frame for plotting
results_df <- data.frame(variable = names(coefficients_of_interest),
                         coefficient = coefficients_of_interest,
                         p_value = p_values_of_interest,
                         stringsAsFactors = FALSE)

# Sort by coefficient value for better visualization
results_df <- results_df[order(abs(results_df$coefficient), decreasing = TRUE),]

# Plot
ggplot(results_df, aes(x = reorder(variable, coefficient), y = coefficient, fill = p_value < 0.001)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = sprintf("%.7f", coefficient)), vjust = -0.5, size = 3) +
  coord_flip() +
  labs(x = "Variables", y = "Coefficient Estimate",
       title = "Top 5 significant features (p < 0.1)",
       subtitle = "High statistical significance (p < 0.001) indicated by color") +
  scale_fill_manual(values = c("TRUE" = "lightblue", "FALSE" = "gray")) +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 10))

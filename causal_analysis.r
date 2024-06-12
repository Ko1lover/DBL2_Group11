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

# Set current working directory
print(utils::getSrcDirectory(function(){}))
print(utils::getSrcFilename(function(){}, full.names = TRUE))
# Run this for RStudio
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
# # Run this if not RStudio
# setwd(getSrcDirectory(function(){})[1])

# ------------------------------------------------------------------------------

# TRUST CAUSAL ANALYSIS

# DATA PROCESSING ----------------------------------------------

# Load the dataset
agg_data <- read_csv("FINAL_agg_Dataset.csv")

# Create a time variable based on year and quarter, arrange data, convert to pdata.frame, and remove year and quarter columns
data_trust <- agg_data %>%
  mutate(time = as.Date(paste0(year, "-", (quarter - 1) * 3 + 1, "-01"), format = "%Y-%m-%d")) %>%
  arrange(borough, time) %>%
  pdata.frame(index = c("borough", "time")) %>%
  subset(select = -c(year, quarter, GoodJoblocal, total_q_crimes))

# Convert to data frame
df <- as.data.frame(data_trust, index = c("borough", "time"))

# Data descriptors
describe(df)
summary(df)


# MODELS -------------------------------------------------------

# Create the model formula for trust
factors_t <- c(colnames(df[, -which(names(df) %in% c("TrustMPS"))]))
# print(factors_t)  # Factors for Trust
eq_t <- as.formula(paste("TrustMPS ~ ", paste(factors_t, collapse="+")))
# print(eq_t)

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
                   + other_q_crimes
)


# Fit a linear model to check for multicollinearity
lm_model <- lm(eq_t, data = df)

# Identify and list aliased (perfectly collinear) coefficients
aliased <- alias(lm_model)$Complete

if (length(aliased) > 0) {
  aliased_vars <- names(which(apply(aliased, 2, function(x) any(x != 0))))
  cat(paste("Aliased variables detected: \n", paste(aliased_vars, collapse="\n"), "\n"))

  df <- df %>% select(-all_of(aliased_vars))

  # Update model formula
  factors_t <- colnames(d[, -which(names(df) %in% c("TrustMPS"))])
  eq_t <- as.formula(paste("TrustMPS ~ ", paste(factors_t, collapse="+")))

  # Re-fit the linear model
  lm_model <- lm(eq_t, data = df)

} else {
  print("No aliased variables detected.")
}


# VIF > 5~10 are highly multicollinear and have to be dealt with
vif_values <- vif(lm_model)
print(vif_values)


# Fit the Pooled OLS model
pool_model <- plm(eq_t, index = c("borough", "time"), data = df, model = "pooling")
# summary(pool_model)


# Fit models with time effects, entity effects, and mixed effects
pool_entity_eff <- plm(update(eq_t, reformulate(c(".", "borough"))), index = c("borough", "time"), data = df, model = "pooling")
pool_mixed_eff <- plm(update(eq_t, reformulate(c(".", "borough + time"))), index = c("borough", "time"), data = df, model = "pooling")
pool_time_eff <- plm(update(eq_t, reformulate(c(".", "time"))), index = c("borough", "time"), data = df, model = "pooling")

# summary(pool_entity_eff)
# summary(pool_mixed_eff)
# summary(pool_time_eff)


# TRUE would suggest there are significant differences in the intercepts across different entities and/or time that are not captured by the pooled OLS model
sprintf("Fixed entity effect in data: %s", pFtest(pool_entity_eff, pool_model)$p.value <= 0.05)
sprintf("Fixed mixed effect in data: %s", pFtest(pool_mixed_eff, pool_model)$p.value <= 0.05)
sprintf("Fixed time effect in data: %s", pFtest(pool_time_eff, pool_model)$p.value <= 0.05)

# RESULT: First two tests return TRUE which indicates Entity effects and Time effects are not accounted for in the OLS model

# Testing for presence of random entity (borough) and time effects (Breush-Pagan 1980)
sprintf("Random entity effect in data: %s", plmtest(pool_model, type="bp", effect="individual")$p.value <= 0.05)
sprintf("Random time effect in data: %s", plmtest(pool_model, type="bp", effect="time")$p.value<= 0.05)
sprintf("Random mixed effect in data: %s", plmtest(pool_model, type="ghm", effect="twoways")$p.value<= 0.05) # ghm is only available for twoways, but robust for unbalanced panel


# Fixed effects model - accounts for
fe_model <- plm(eq_t, index = c("borough", "time"), data = df, model = "within")
summary(fe_model)

# Random effects model
re_model <- plm(eq_t, index = c("borough", "time"), data = df, model = "random", random.method = "walhus")
summary(re_model)


# Hausman test:
#  The null hypothesis is that the preferred model is random effects;
#  the alternative hypothesis is that the preferred model is fixed effects.
h_test <- phtest(fe_model, re_model)
summary(h_test)
# p-value is 1 which is >0.05 so we don't reject alternative hypothesis. FE is preferred.


# Test for autocorrelation:
#  The null hypothesis is that there is no autocorrelation;
#  the alternative hypothesis is that there is no autocorrelation but serial correlation (we reject this if p<0.05).
auto_corr_test <- pbgtest(fe_model)
print(auto_corr_test$p.value)


# Test for heteroskedasticity:
#  The null hypothesis is that there is homoskedasticity;
#  the alternative hypothesis is that there heteroskedasticity (reject this if p<0.05).
het_test <- bptest(eq_t, data=df, studentize=F)
print(het_test$p.value)


# # Robust standard errors to ensure correct output regardless of heteroskedasticity and autocorrelation
# # More robust and reliable
# coeftest(fe_model, vcovHC(fe_model, method= "arellano"))


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

# Create the model formula for trust
factors_t <- c(colnames(df[, -which(names(df) %in% c("GoodJoblocal"))]))
print(factors_t)  # Factors for Trust
eq_t <- as.formula(paste("GoodJoblocal ~ ", paste(factors_t, collapse="+")))
# print(eq_t)

eq_t <- as.formula(GoodJoblocal ~
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
                   + other_q_crimes
                   + time
)


# Fit a linear model to check for multicollinearity
lm_model <- lm(eq_t, data = df)

# Identify and list aliased (perfectly collinear) coefficients
aliased <- alias(lm_model)$Complete

if (length(aliased) > 0) {
  aliased_vars <- names(which(apply(aliased, 2, function(x) any(x != 0))))
  cat(paste("Aliased variables detected: \n", paste(aliased_vars, collapse="\n"), "\n"))

  df <- df %>% select(-all_of(aliased_vars))

  # Update model formula
  factors_t <- colnames(d[, -which(names(df) %in% c("GoodJobLocal"))])
  eq_t <- as.formula(paste("GoodJoblocal ~ ", paste(factors_t, collapse="+")))

  # Re-fit the linear model
  lm_model <- lm(eq_t, data = df)

} else {
  print("No aliased variables detected.")
}


# VIF > 5~10 are highly multicollinear and have to be dealt with
vif_values <- vif(lm_model)
print(vif_values)


# Fit the Pooled OLS model
pool_model <- plm(eq_t, index = c("borough", "time"), data = df, model = "pooling")
summary(pool_model)


# Fit models with time effects, entity effects, and mixed effects
pool_entity_eff <- plm(update(eq_t, reformulate(c(".", "borough"))), index = c("borough", "time"), data = df, model = "pooling")
pool_mixed_eff <- plm(update(eq_t, reformulate(c(".", "borough + time"))), index = c("borough", "time"), data = df, model = "pooling")
pool_time_eff <- plm(update(eq_t, reformulate(c(".", "time"))), index = c("borough", "time"), data = df, model = "pooling")

summary(pool_entity_eff)
summary(pool_mixed_eff)
summary(pool_time_eff)


# TRUE would suggest there are significant differences in the intercepts across different entities and/or time that are not captured by the pooled OLS model
sprintf("Fixed entity effect in data: %s", pFtest(pool_entity_eff, pool_model)$p.value <= 0.05)
sprintf("Fixed mixed effect in data: %s", pFtest(pool_mixed_eff, pool_model)$p.value <= 0.05)
sprintf("Fixed time effect in data: %s", pFtest(pool_time_eff, pool_model)$p.value <= 0.05)

# RESULT: First two tests return TRUE which indicates Entity effects and Time effects are not accounted for in the OLS model

# Testing for presence of random entity (borough) and time effects (Breush-Pagan 1980)
sprintf("Random entity effect in data: %s", plmtest(pool_model, type="bp", effect="individual")$p.value <= 0.05)
sprintf("Random time effect in data: %s", plmtest(pool_model, type="bp", effect="time")$p.value<= 0.05)
sprintf("Random mixed effect in data: %s", plmtest(pool_model, type="ghm", effect="twoways")$p.value<= 0.05) # ghm is only available for twoways, but robust for unbalanced panel


# Fixed effects model - accounts for
fe_model <- plm(eq_t, index = c("borough", "time"), data = df, model = "within")
summary(fe_model)

# Random effects model
re_model <- plm(eq_t, index = c("borough", "time"), data = df, model = "random", random.method = "walhus")
summary(re_model)


# Hausman test:
#  The null hypothesis is that the preferred model is random effects;
#  the alternative hypothesis is that the preferred model is fixed effects.
h_test <- phtest(fe_model, re_model)
summary(h_test)
# p-value is 0.90 which is >0.05 so we don't reject alternative hypothesis. FE is preferred.


# Test for autocorrelation:
#  The null hypothesis is that there is no autocorrelation;
#  the alternative hypothesis is that there is no autocorrelation but serial correlation (we reject this if p<0.05).
auto_corr_test <- pbgtest(fe_model)
print(auto_corr_test$p.value)


# Test for heteroskedasticity:
#  The null hypothesis is that there is homoskedasticity;
#  the alternative hypothesis is that there heteroskedasticity (reject this if p<0.05).
het_test <- bptest(eq_t, data=df, studentize=F)
print(het_test$p.value)

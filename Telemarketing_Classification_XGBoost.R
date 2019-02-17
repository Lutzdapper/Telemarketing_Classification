######################## Fallstudie zum Telemarketing einer Bank ########################################
################################  Data Mining  ############################################
## Leeren der Umgebung
rm(list = ls())

# Laden von notwendigen Paketen
library(dplyr)
library(lubridate)
library(readr)

## Einlesen der Bankdaten
load("/home/.../DataMining/ProjektTelemarketing/Bank_data.RData")

# Summary des Datensatzes
summary(df_bank)

# Pruefen auf fehlende Werte
any(is.na(df_bank))

# Haeufigkeiten Anzahl Tage seit letzter Kampagne
table(df_bank$pdays)

# Anteil der Personen, die schon ueber 3 Jahre nicht mehr in einer Kampagne beruecksichtig wurden an der Gesamtzahl
38530/40000

# Haeufigkeiten der Anzahl der Angestellten
table(df_bank$nr.employed)

# Haeufigkeitsverteilung der Variable Alter
table(df_bank$age)

# Aureißeranalyse
#boxplot(df_bank[,2:length(df_bank)], main ="Boxplot-Analyse", las = 2)

boxplot(df_bank[,"age"], main ="Boxplot Alter Kunden")
boxplot(df_bank[,"pdays"], main ="Boxplot Tage seit Kampagne")
# pdays Ausreißer könnten NA-Werte sein bzw. unsere Annahme: die letzte Kampagne ist schon knapp 3 Jahre her
boxplot(df_bank[,"previous"], main ="Boxplot Anzahl Kontaktversuche")
boxplot(df_bank[,"emp.var.rate"], main ="Boxplot Veränderung der Erwerbstätigenquote")
boxplot(df_bank[,"cons.price.idx"], main ="Boxplot Konsumentenpreisindex")
boxplot(df_bank[,"cons.conf.idx"], main ="Boxplot Verbrauchervertrauensindex")
boxplot(df_bank[,"euribor3m"], main ="Boxplot Wert des Euribor")
boxplot(df_bank[,"nr.employed"], main ="Boxplot Anzahl Mitarbeiter")

# Struktur des Datensatzes
str(df_bank)

## Korrelation der metrischen wirtschaftlichen Variablen
cor(df_bank[c("emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m")],
    use = "pairwise")

###################################### Entscheidungsbaeume ##############################################
library(rpart)
dt_model <- rpart(formula = y ~ age + job + marital + education + default + 
                    housing + loan + contact + month + day_of_week + pdays + previous + poutcome +
                    emp.var.rate + cons.price.idx + cons.conf.idx + euribor3m,
                  data = df_bank, method = "class", 
                  control = rpart.control(cp = 0.0001))
printcp(dt_model)
plotcp(dt_model)

# Optimierung des Komplexitaetsfaktors
dt_model <- rpart(formula = y ~ age + job + marital + education + default + 
                    housing + loan + contact + month + day_of_week + pdays + previous + poutcome +
                    emp.var.rate + cons.price.idx + cons.conf.idx + euribor3m,
                  data = df_bank, method = "class", 
                  control = rpart.control(cp = 0.0024455))
printcp(dt_model)
plotcp(dt_model)


# Entscheidungsbaum visuell darstellen
plot(dt_model, main = "Entscheidungsbaum")
text(dt_model, cex = 0.4)

# Wichtigkeit der Variablen
barplot(dt_model$variable.importance, main ="Wichtigkeit der Variablen (Entscheidungsbaum)",horiz = TRUE, las = 1, cex.names = 0.5)


################################# Erstellen des Vorhersagemodells ######################################

# Input zur Berechnung einer Zufallszahl festlegen (damit der Datensatz immer identisch bleibt)
set.seed(123)

# Groeße des Trainingsdatensatzes festlegen (hier 70%)
train_share <- 0.7

# Kunden aufteilen in Trainings- und Testdatensatz
sample_size <- nrow(df_bank)
sample_train <- sample(x = 1:sample_size, size = floor(sample_size*train_share), 
                       replace = FALSE)
sample_test <- setdiff(1:sample_size, sample_train)

# Trainingsdatensatz und Testdatensatz definieren
df_train <- df_bank[sample_train,]
df_test <- df_bank[sample_test,]

# Datensaetze (x) mit allen unabhaenigigen Variablen zum Trainieren der abhaengigen Variable y
df_train_x <- df_train[,-which(names(df_train) %in% "y")]
df_train_y <- factor(df_train$y)

df_test_x <- df_test[,-which(names(df_test) %in% "y")]
df_test_y <- factor(df_test$y)

##################################### Random Forest ####################################################

# Pakete laden
library(randomForest)

rf_model <- randomForest(x = df_train_x, y = df_train_y,
                         xtest = df_test_x, ytest = df_test_y,
                         ntree = 100, do.trace = 10, importance = TRUE)
rf_model

# mtry tunen
tune_mtry <- tuneRF(x = df_train_x, y = df_train_y, mtryStart = 6)

# randomForest Modell mit dem neuen mtry
rf_model <- randomForest(x = df_train_x, y = df_train_y,
                         xtest = df_test_x, ytest = df_test_y,
                         ntree = 100, do.trace = 10, mtry = 3, importance = TRUE)
rf_model
plot(rf_model, main = "Random Forest Modell")

## Visualisierung der Wichtigkeit der Variablen
rf_model$importance
varImpPlot(rf_model, main= "Wichtigkeit der Variablen (Random Forest)")

######################################## Random Forest 2 ##############################################

# dieser RandomForest arbeitet mit Wahrscheinlichkeiten oder?
rf_model_2 <- randomForest(x = df_train_x, y = df_train_y,
                           xtest = df_test_x, ytest = df_test_y, 
                           do.trace = 20, cutoff = c(0.95,0.05))



######################################## Ada Boosting  ################################################
library(ada)
# Ada-Boost-Modell trainieren
# ada_model <- ada(x = df_train_x, y = df_train_y, 
#                 test.x = df_test_x, test.y = df_test_y, 
#                 loss = "exponential", type = "discrete", iter = 100)
ada_model <- ada(y ~ age + job + marital + education + default + 
                   housing + loan + contact + month + day_of_week + pdays + previous + poutcome +
                   emp.var.rate + cons.price.idx + cons.conf.idx + euribor3m + nr.employed,
                 data = df_train,  test.x = df_test_x, test.y = df_test_y, 
                 loss = "exponential", type = "discrete", iter = 100)

plot(ada_model)


########################################### XG - Boosting  #############################################
library(xgboost)

# Trainieren des Modells mit den jeweiligen Datensaetzen
xgb_df_train <- xgb.DMatrix(model.matrix(object = y ~ age + job + marital + education + default + 
                                           housing + loan + contact + month + day_of_week + pdays + previous + poutcome +
                                           cons.conf.idx  + nr.employed + emp.var.rate + cons.price.idx + euribor3m, data = df_train), label = ifelse(df_train_y=="yes", 1, 0))
xgb_df_test <- xgb.DMatrix(model.matrix(object = y ~ age + job + marital + education + default + 
                                          housing + loan + contact + month + day_of_week + pdays + previous + poutcome +
                                          cons.conf.idx  + nr.employed + emp.var.rate + cons.price.idx + euribor3m, data = df_test), label = ifelse(df_test_y=="yes", 1, 0))
# Einstellen der Parameter
param <- list(max_depth = 6, eta = 0.05, silent = 1, nthread = 2, 
              objective = "binary:logistic", eval_metric = "logloss")
watchlist <- list(train = xgb_df_train, eval = xgb_df_test)

# Vorhersagemodell gemaeß XG- Boosting
xgb_model <- xgb.train(param, xgb_df_train, nrounds = 121, watchlist = watchlist )


# cv <- xgb.cv(params = param,
#        data = xgb_df_train,
#        nrounds = 1000,
#        watchlist = watchlist,
#        nfold = 5,
#        showsd = T,
#        metrics = "logloss",
#        stratified = T,
#        print_every_n = 10,
#        early_stop_round = 50,
#        maximize = F)
# 
# library(dplyr)
# cv$evaluation_log %>%
#   filter(test_error_mean == min(test_error_mean))
# #
# min_logloss = min(cv$evaluation_log$test_logloss_mean)
# min_logloss
# min_logloss_index = which.min(cv$evaluation_log$test_logloss_mean)
# min_logloss_index

############################################# GBM ######################################################
library(gbm)
# Train GBM model
df_train$y <- ifelse(df_train$y == "yes", 1, 0)
gbm_model <- gbm(formula = y ~  age + job + marital + education + default + 
                   housing + loan + contact + month + day_of_week + pdays + previous + poutcome +
                   cons.conf.idx  + nr.employed + emp.var.rate + cons.price.idx + euribor3m, data = df_train,
                 n.trees = 70,
                 distribution = "bernoulli")


######################################## Prediction ###################################################

# Vorhersage fuer die Datensaetze erstellen
pred_train_xgb <- predict(xgb_model, newdata = xgb_df_train)
pred_test_xgb <- predict(xgb_model, newdata = xgb_df_test)

pred_test_dt <- predict(dt_model, newdata = df_test_x, type = "prob")
pred_test_rf <- rf_model$test$votes
pred_test_rf_2 <- rf_model_2$test$votes
pred_test_ada <- predict(ada_model, newdata = df_test_x, type = "probs")
pred_test_gbm <- predict(gbm_model, n.trees = gbm_model$n.trees, newdata = df_test_x, type = "response")



#################################### Performance measurement ############################################
#---------------------------------------------------------------------------------------------------------------------------------------

# Berechnung der Vorhersagen
library(ROCR)
prediction_test_dt <- prediction(pred_test_dt[,2], labels = df_test$y)
prediction_test_rf <- prediction(pred_test_rf[,2], labels = df_test$y)
prediction_test_rf_2 <- prediction(pred_test_rf_2[,2], labels = df_test$y)
prediction_test_ada <- prediction(pred_test_ada[,2], labels = df_test$y)

df_test$y <- ifelse(df_test$y == "yes", 1, 0)
prediction_train_xgb <- prediction(pred_train_xgb, labels = df_train$y)
prediction_test_xgb <- prediction(pred_test_xgb, labels = df_test$y)
prediction_test_gbm <- prediction(pred_test_gbm, labels = df_test$y)

# ROC berechnen
roc_train_xgb <- ROCR::performance(prediction_train_xgb, measure="tpr", x.measure="fpr")
roc_test_xgb <- ROCR::performance(prediction_test_xgb, measure="tpr", x.measure="fpr")

roc_test_dt <- ROCR::performance(prediction_test_dt, measure="tpr", x.measure="fpr")
roc_test_rf <- ROCR::performance(prediction_test_rf, measure="tpr", x.measure="fpr")
roc_test_rf_2 <- ROCR::performance(prediction_test_rf_2, measure="tpr", x.measure="fpr")
roc_test_ada <- ROCR::performance(prediction_test_ada, measure="tpr", x.measure="fpr")
roc_test_gbm <- ROCR::performance(prediction_test_gbm, measure="tpr", x.measure="fpr")

# AUC berechnen
auc_train_xgb <- ROCR::performance(prediction_train_xgb, measure="auc")
auc_test_xgb <- ROCR::performance(prediction_test_xgb, measure="auc")

auc_test_dt <- ROCR::performance(prediction_test_dt, measure="auc")
auc_test_rf <- ROCR::performance(prediction_test_rf, measure="auc")
auc_test_rf_2 <- ROCR::performance(prediction_test_rf_2, measure="auc")
auc_test_ada <- ROCR::performance(prediction_test_ada, measure="auc")
auc_test_gbm <- ROCR::performance(prediction_test_gbm, measure="auc")


# Den y-Wert der AUC-Berechnung aufrunden
auc_train_xgb <- round(auc_train_xgb@y.values[[1]], 4)
auc_test_xgb <- round(auc_test_xgb@y.values[[1]], 4)

# Ausgabe des AUC-Wertes
auc_train_xgb
auc_test_xgb

auc_test_dt <- round(auc_test_dt@y.values[[1]], 4)
auc_test_rf <- round(auc_test_rf@y.values[[1]], 4)
auc_test_rf_2 <- round(auc_test_rf_2@y.values[[1]], 4)
auc_test_ada <- round(auc_test_ada@y.values[[1]], 4)
auc_test_gbm <- round(auc_test_gbm@y.values[[1]], 4)


# Plotten ROC
plot(roc_train_xgb, main = "ROC Vergleich")
plot(roc_test_xgb, add = TRUE, col="red")
plot(roc_test_dt, add = TRUE, col = "orange", lty =2)
plot(roc_test_rf, add = TRUE, col = "purple", lty =2)
plot(roc_test_rf_2, add = TRUE, col = "yellow", lty =2)
plot(roc_test_ada, add = TRUE, col = "blue", lty =2)
plot(roc_test_gbm, add = TRUE, col = "green", lty =2)
abline(0,1, col = "grey", lty = 2)
legend("bottomright", legend = c(paste0("(XGB Train AUC = ", auc_train_xgb, ")"),
                                 paste0("(XGB Test AUC = ", auc_test_xgb,")"),
                                 paste0("(DT Test AUC = ", auc_test_dt, ")"),
                                 paste0("(RF Test AUC = ", auc_test_rf,")"),
                                 paste0("(RF 2 Test AUC = ", auc_test_rf_2,")"),
                                 paste0("(Ada Test AUC = ", auc_test_ada,")"),
                                 paste0("(GBM Test AUC = ", auc_test_gbm,")")), 
       col =c("black", "red", "orange", "purple", "yellow", "blue", "green") ,lty = 1)

## XGB - Boosting ist das beste Modell

########################## Kostenoptimales Entscheidungskriterium fuer Auswahl ###########################


# Aufstellen der Kostenmatrix gemaeß der Aufgabenstellung
costs <- matrix(c(0,-53,-10,70), ncol = 2, byrow = TRUE)
rownames(costs) <- colnames(costs) <- c(FALSE, TRUE)
costs <- as.table(costs)
names(dimnames(costs)) <- c("Prediction", "Reality")
costs

# Funktion erstellen zur Berechnung des Deckungsbeitrags fuer verschiedene Schwellenwerte
costf <- function(x){
  cmx <- table(pred_test_xgb>x, df_test$y, dnn = c("Prediction", "Reality"))
  return(sum(cmx*costs))
}

# Berechnung der DB-Werte je Schwellenwert
s <- seq(0.05,0.7, 0.001)
r <- numeric(0)
for (i in s){
  r <- c(r,costf(i))
}

# Plot DB-Hoehe je Schwellenwert
plot(s,r,type = "l", main ="Deckungsbeitrag je Schwellenwert", xlab = "Threshold", ylab = "Total Margin")
s[which.max(r)] # s = 0,085; der Wert wird bei der Confusion Matrix eingesetzt

# Prozesswwiederholung zur Darstellung des Gewinnverlaufs im positiven Bereich
s <- seq(0.04,0.3, 0.001)
r <- numeric(0)
for (i in s){
  r <- c(r,costf(i))
}
plot(s,r,type = "l", main ="Deckungsbeitrag je Schwellenwert", xlab = "Threshold", ylab = "Total Margin")

# Confusion Matrix
cm2 <- table(pred_test_xgb>0.085, df_test$y, dnn = c("Prediction", "Reality"))
cm2

# Deckungsbeitrag berechnen
sum(cm2*costs) # 23001 ist max. DB

# Sensitivitaet (Sensitivity): Anzahl der richtig vorhergesagten, die auf die Kampagne eingehen
# zu allen, die auf die Kampagne eingehen wuerden (ca. 28%)
cm2[2,2] / sum(cm2[2,])

# Spezifitaet (Specificity):Anzahl der richtig vorhergesagten, die nicht auf die Kampagne eingehen
# zu allen, die nicht auf die Kampagne eingehen (ca. 95%)
cm2[1,1] / sum(cm2[1,])

# Genauigkeit (Accuracy): Anzahl der richtig vorhergesagten zu der Gesamtmenge (ca. 76%)
(cm2[1,1] + cm2[2,2]) / sum(cm2)



############################### Vorhersage fuer den neuen Datensatz ######################################

# Den Datensatz derzu bestimmenden Kunden einlesen
load("/home/.../DataMining/ProjektTelemarketing/Bank_test_data.RData")

#Struktur des Datensatzes anzeigen (Variable y nicht gegeben)
str(df_bank_pred)

# Neue Variable y einfuegen mit Dummywerte = 1
df_bank_pred <- cbind(df_bank_pred, y = 1)

# Umwandeln der Matrix in eine xgb.Matrix
xgb_df_bank_pred <- xgb.DMatrix(model.matrix(object = y ~ age + job + marital + education + default + 
                                               housing + loan + contact + month + day_of_week + pdays 
                                             + previous + poutcome + cons.conf.idx  + nr.employed + 
                                               emp.var.rate + cons.price.idx + euribor3m, data = df_bank_pred))

# Vorhersage gemaeß dem ausgewaehlten (besten) Vorhersagemodell
pred_xgb_final <- predict(xgb_model, newdata = xgb_df_bank_pred)

# Definieren, ab welcher Wahrscheinlichkeit ein Kunde angerufen werden soll
xgb_df_bank_pred <- ifelse(pred_xgb_final > 0.085, "yes", "no")

# Speichern des Dataframe in der Variable "pred"
pred <- as.data.frame(cbind(df_bank_pred["id"], xgb_df_bank_pred))

str(pred)
summary(pred)
## 72 Kunden werden angerufen, 1.116 Kunden werden nicht angerufen
72/(72+1116) # ca. 6% der 1.188 Kunden werden nur angerufen

# Maximaler DB bei der Prognose
72*70

# Speichern des Dataframe pred  
save(pred, file="Customers_scored.RData")

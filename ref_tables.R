temptable = subset(labtable, 
                   type1 == "Words" | type2 == "Words")
library(reshape)

long <- melt(temptable[ , c(19, 20, 25:74)],
             id = c("Full.Reference",
                    "language", "notes_lang"))

long <-subset(long, value > 0)
write.csv(long, 
          "language_long.csv",
          row.names = F)

library(tidyverse)
tabled <- pivot_wider(data = as.data.frame(table(long$language, long$variable)),
                      id_cols = Var1,
                      names_from = Var2,
                      values_from = Freq)

write.csv(tabled, )
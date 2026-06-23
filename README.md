# Bh-tsne
L'objectif de ce projet est d'implementer l'algorithme barnes-hut t-sne.
La fonction t-sne permet de choisir entre plusieurs manières de calcul(Avec ou sans Barnes-hiut, avec ou sans parallelisme)

Le paramètre perplexity est lié au nombre de voisins moyen dans l'espace d'arrivée. Elle dépends du nombre de point donné en paramètre. Tester plusieurs valeurs peut être interessant(augmenter le parametre si les points sont trop éloignés, le diminuer si ils sont trop proches)  
Il est conseillé d'avoir un nombre d'iteration total (max_iter) qui soit au moins le double de early_exaggeration_iter.
Au delà de theta = 0.8, l'approximation donnée par l'algorithme de Barnes-Hut ne produit souvent plus de résultats satisfaisant.

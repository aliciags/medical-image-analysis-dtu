# How to pull changes from main to another branch

``` bash
git checkout main
git pull
git checkout <branch-name>
git merge main
# EDIT CODE
# IF NEW FILES CREATED
git add .
# IF NO NEW FILES DIRECTLY
git commit -a -m “message”
git push
# CHECK EVERYTHING UPLOADED CORRECTLY
git status
# THEN FROM GITHUB CREATE A PULL REQUEST TO MAIN
```
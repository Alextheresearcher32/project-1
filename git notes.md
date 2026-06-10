git notes.md
# basic commands
initialize a repository: git init 
clone a repository: git clone <repository-url>
check status: git status 

# working with commits 
add files to staging: git add <file>
git add. 
commit changes: git commit -m "commit message" 
view commit history: git log

#branching 
create branch: git branch <branch name >
switch branches: git checkout <branch-name>
create and switch: git checkout -b <branch name>

# merging and rebasing
merge branches: git merge <branch name>
rebase branches: git rebase <branch-name>

# remote repository
add remote: git remote add origin <repository-url>
view remotes: git remote -v
push to remote: git push origin <branch-name>
pull from remote: git pull

#stashing and cleaning 
stash changes: git stash 
apply stash: git stash apply

# additional commands 
show differences: git diff
remove files: git rm <file>
rename file: git mv <old name> <new name>

local repository (personal)
remote repository (share collab)
type changes commit in local repository then push to remote repository

cd (change directory)
mk dir (make directory)
git init (init repository)
cls (clear)
If i wanna add commit type git add then file name then git status to check

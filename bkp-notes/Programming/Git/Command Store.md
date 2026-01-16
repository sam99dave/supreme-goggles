
```bash
# Git Log
git log # -- Interactive
git --no-pager log -n 10 # -- Non interactive + limit

# Git Cat
git cat-file -p <hash | tree | blob> # -- See contents

# Git Config
git config --add --global user.name "" # Omit --global for local/repo lvl setting
git config --add --global user.email ""
git config --list --local # To view the config
git config --get <key> # To get single key value | key format -> <section>.<keyname>
git config --unset <key> # Remove keys (use --local/--global accordingly)
# Git config can have duplicate keys (to remove all ot once)
git config --unset-all <key>
git config --remove-section section
```

There are several locations where Git can be configured. From more general to more specific, they are:

- **system**: /etc/gitconfig, a file that configures Git for all users on the system
- **global**: ~/.gitconfig, a file that configures Git for all projects of a user
- **local**: .git/config, a file that configures Git for a specific project
- **worktree**: .git/config.worktree, a file that configures Git for part of a project




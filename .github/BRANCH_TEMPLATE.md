See [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)

![Git Model](https://nvie.com/img/git-model@2x.png)

|Branch Name| From | Merge into |Description|
|:-:|:-|
| `master`  |  |  | You put the actual product files, tag them when you `release` them. |
| `develop` | `master` , `release` | `master` | Development branch. It's the latest branch before `release` and will be merged into the `master` branch when the `release` is ready. |
| `feature` | `develop` | `develop` | A development branch for additional features and bug fixes. Branch from the `develop` branch and merge it into the `develop` branch after modifying the source. |
| `release` (`documentation`) | `develop` | `master` , `develop` | A branch that prepares and fine-tunes before `release`. Branch from the `develop` branch and tag it (Figure: 1.0)  |  |. After adjustment, merge it into the `master` branch. Then merge it into your `develop` branch. |
| `hot-fix` | `master` | `master` | A branch for emergency corrections in the `master` branch. Branch from the `master` branch, merge into the `master` branch and tag it. |

---
layout: default
title: Home
nav_order: 1
description: "LEG(Linearly Estimated Gradient) is a high level deep learning model explainer."
permalink: /
---

# LEG(Linearly Estimated Gradient)
{: .fs-9 }

LEG gives you a visual interpretation of what is happening under the hood of deep learning models.
{: .fs-6 .fw-300 }

[Get started now](#getting-started){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 } [View it on GitHub](https://github.com/Paradise1008/LEG){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## Getting started

### Dependencies

* [cvxpy](https://github.com/cvxgrp/cvxpy) 
* [Mosek](https://www.mosek.com/documentation/)
* [tensorflow/keras](https://www.tensorflow.org/)
* [matplotlib](https://matplotlib.org/users/installing.html)
* [skimage](https://github.com/scikit-image/scikit-image)

### [Quick start]({{ site.baseurl }}/docs/quick%20start)

1. Add Just the Docs to your Jekyll site's `_config.yml` as a [remote theme](https://blog.github.com/2017-11-29-use-any-theme-with-github-pages/)

```yaml
remote_theme: just-the-docs/just-the-docs
```

<small>You must have GitHub Pages enabled on your repo, one or more Markdown files, and a `_config.yml` file. [See an example repository](https://github.com/pmarsceill/jtd-remote)</small>

### Local installation: Use the gem-based theme

1. Install the Ruby Gem
  ```bash
  $ gem install just-the-docs
  ```
  ```yaml
  # .. or add it to your your Jekyll site’s Gemfile
  gem "just-the-docs"
  ```

2. Add Just the Docs to your Jekyll site’s `_config.yml`
  ```yaml
  theme: "just-the-docs"
  ```

3. _Optional:_ Initialize search data (creates `search-data.json`)
  ```bash
  $ bundle exec just-the-docs rake search:init
  ```

3. Run you local Jekyll server
  ```bash
  $ jekyll serve
  ```
  ```bash
  # .. or if you're using a Gemfile (bundler)
  $ bundle exec jekyll serve
  ```

4. Point your web browser to [http://localhost:4000](http://localhost:4000)

If you're hosting your site on GitHub Pages, [set up GitHub Pages and Jekyll locally](https://help.github.com/en/articles/setting-up-your-github-pages-site-locally-with-jekyll) so that you can more easily work in your development environment.

### Configure Just the Docs



---

## About the project

LEG explainer 2019-{{ "now" | date: "%Y" }} by [Paradise1008](https://github.com/Paradise1008).

## License

None


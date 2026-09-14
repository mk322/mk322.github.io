(function () {
  "use strict";
  var search = document.getElementById("blog-search");
  var tag = document.getElementById("blog-tag");
  if (!search || !tag) return;
  var cards = Array.from(document.querySelectorAll(".blog-card"));
  var posts = cards.map(function (card) {
    return { element: card, text: card.textContent.toLowerCase(), tags: JSON.parse(card.dataset.tags) };
  });
  var params = new URLSearchParams(window.location.search);
  search.value = params.get("q") || "";
  var initialTag = params.get("tag") || "";
  if (Array.from(tag.options).some(function (option) { return option.value === initialTag; })) tag.value = initialTag;
  function filter() {
    var query = search.value.trim().toLowerCase();
    var count = 0;
    posts.forEach(function (post) {
      var matches = (!query || post.text.includes(query)) && (!tag.value || post.tags.includes(tag.value));
      post.element.hidden = !matches;
      if (matches) count++;
    });
    document.getElementById("blog-results").textContent = count + (count === 1 ? " post" : " posts");
    document.getElementById("blog-no-results").hidden = count !== 0;
  }
  document.querySelector(".blog-filters").hidden = false;
  search.addEventListener("input", filter);
  tag.addEventListener("change", filter);
  filter();
})();

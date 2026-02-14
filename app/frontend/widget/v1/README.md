# TMU Arts Chatbot Widget (v1)

Static assets served by the FastAPI service.

## URLs (when API is running)
- Script: `/widget/v1/widget.js`
- Demo: `/widget/v1/demo.html`

## Embed (script)

```html
<div id="tmu-chat"></div>
<script src="https://YOUR_API_DOMAIN/widget/v1/widget.js"></script>
<script>
  window.TMUChatbot.init({
    container: "#tmu-chat",
    apiBaseUrl: "https://YOUR_API_DOMAIN",
    mode: "public",            // or "admin"
    title: "TMU Arts Chat",
    enableCitations: true,
    enableDebug: false,
    defaultParams: { top_k: 6, num_candidates: 20 } // used for admin mode
  });
</script>
```

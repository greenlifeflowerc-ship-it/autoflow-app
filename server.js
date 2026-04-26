import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import axios from "axios";
import cron from "node-cron";
import { GoogleGenerativeAI } from "@google/generative-ai";

dotenv.config();

const app = express();

app.use(cors());
app.use(express.json({ limit: "25mb" }));

const PORT = process.env.PORT || 3000;
const GRAPH_VERSION = "v19.0";

const META_ACCESS_TOKEN = process.env.META_ACCESS_TOKEN;
const IG_USER_ID = process.env.IG_USER_ID;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

const posts = [];

function requireMetaConfig() {
  if (!META_ACCESS_TOKEN || !IG_USER_ID) {
    throw new Error("Missing META_ACCESS_TOKEN or IG_USER_ID in environment variables.");
  }
}

function requireGeminiConfig() {
  if (!GEMINI_API_KEY) {
    throw new Error("Missing GEMINI_API_KEY in environment variables.");
  }
}

function isPublicUrl(url) {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "https:" || parsed.protocol === "http:";
  } catch {
    return false;
  }
}

async function publishToInstagram({ imageUrl, caption }) {
  requireMetaConfig();

  if (!imageUrl || !isPublicUrl(imageUrl)) {
    throw new Error("imageUrl must be a public direct URL.");
  }

  const createContainerUrl = `https://graph.facebook.com/${GRAPH_VERSION}/${IG_USER_ID}/media`;

  const containerResponse = await axios.post(createContainerUrl, null, {
    params: {
      image_url: imageUrl,
      caption,
      access_token: META_ACCESS_TOKEN
    }
  });

  const creationId = containerResponse.data.id;

  if (!creationId) {
    throw new Error("Meta did not return creation_id.");
  }

  const publishUrl = `https://graph.facebook.com/${GRAPH_VERSION}/${IG_USER_ID}/media_publish`;

  const publishResponse = await axios.post(publishUrl, null, {
    params: {
      creation_id: creationId,
      access_token: META_ACCESS_TOKEN
    }
  });

  return {
    creationId,
    publishId: publishResponse.data.id
  };
}

async function generateCaptionWithGemini({ imageUrl, language = "arabic", tone = "premium" }) {
  requireGeminiConfig();

  const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
  const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

  const prompt = `
You are a social media marketing assistant for artificial trees, artificial flowers, and luxury decoration products.

Generate:
1. Short Instagram caption
2. 8 relevant hashtags
3. Alt text

Language: ${language}
Tone: ${tone}
Image URL: ${imageUrl}

Return strict JSON only:
{
  "caption": "...",
  "hashtags": ["#tag1", "#tag2"],
  "alt_text": "..."
}
`;

  const result = await model.generateContent(prompt);
  const text = result.response.text();

  try {
    return JSON.parse(text.replace(/```json|```/g, "").trim());
  } catch {
    return {
      caption: text,
      hashtags: [],
      alt_text: ""
    };
  }
}

app.get("/", (req, res) => {
  res.json({
    ok: true,
    app: "AutoFlow Backend",
    status: "running"
  });
});

app.get("/api/health", (req, res) => {
  res.json({
    ok: true,
    timestamp: new Date().toISOString()
  });
});

app.get("/api/meta/test-connection", async (req, res) => {
  try {
    requireMetaConfig();

    const url = `https://graph.facebook.com/${GRAPH_VERSION}/${IG_USER_ID}`;
    const response = await axios.get(url, {
      params: {
        fields: "id,username,name",
        access_token: META_ACCESS_TOKEN
      }
    });

    res.json({
      ok: true,
      account: response.data
    });
  } catch (error) {
    res.status(500).json({
      ok: false,
      error: error.response?.data || error.message
    });
  }
});

app.post("/api/meta/publish-now", async (req, res) => {
  try {
    const { imageUrl, caption } = req.body;

    const result = await publishToInstagram({
      imageUrl,
      caption: caption || ""
    });

    res.json({
      ok: true,
      result
    });
  } catch (error) {
    res.status(500).json({
      ok: false,
      error: error.response?.data || error.message
    });
  }
});

app.post("/api/gemini/generate-caption", async (req, res) => {
  try {
    const { imageUrl, language, tone } = req.body;

    const result = await generateCaptionWithGemini({
      imageUrl,
      language,
      tone
    });

    res.json({
      ok: true,
      result
    });
  } catch (error) {
    res.status(500).json({
      ok: false,
      error: error.message
    });
  }
});

app.post("/api/posts", (req, res) => {
  const {
    imageUrl,
    caption,
    hashtags = [],
    scheduledAt
  } = req.body;

  if (!imageUrl || !scheduledAt) {
    return res.status(400).json({
      ok: false,
      error: "imageUrl and scheduledAt are required."
    });
  }

  const post = {
    id: crypto.randomUUID(),
    imageUrl,
    caption: caption || "",
    hashtags,
    finalText: `${caption || ""}\n${hashtags.join(" ")}`.trim(),
    scheduledAt,
    status: "approved",
    publishAttempts: 0,
    metaContainerId: null,
    metaPublishId: null,
    errorMessage: null,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    publishedAt: null
  };

  posts.push(post);

  res.json({
    ok: true,
    post
  });
});

app.get("/api/posts", (req, res) => {
  res.json({
    ok: true,
    posts
  });
});

app.post("/api/posts/:id/retry", async (req, res) => {
  const post = posts.find((p) => p.id === req.params.id);

  if (!post) {
    return res.status(404).json({
      ok: false,
      error: "Post not found."
    });
  }

  try {
    post.status = "publishing";
    post.publishAttempts += 1;
    post.updatedAt = new Date().toISOString();

    const result = await publishToInstagram({
      imageUrl: post.imageUrl,
      caption: post.finalText
    });

    post.status = "published";
    post.metaContainerId = result.creationId;
    post.metaPublishId = result.publishId;
    post.publishedAt = new Date().toISOString();
    post.updatedAt = new Date().toISOString();

    res.json({
      ok: true,
      post
    });
  } catch (error) {
    post.status = "failed";
    post.errorMessage = JSON.stringify(error.response?.data || error.message);
    post.updatedAt = new Date().toISOString();

    res.status(500).json({
      ok: false,
      post
    });
  }
});

cron.schedule("*/5 * * * *", async () => {
  const now = new Date();

  const duePosts = posts.filter((post) => {
    return (
      post.status === "approved" &&
      new Date(post.scheduledAt) <= now &&
      post.publishAttempts < 3
    );
  });

  for (const post of duePosts) {
    try {
      post.status = "publishing";
      post.publishAttempts += 1;
      post.updatedAt = new Date().toISOString();

      const result = await publishToInstagram({
        imageUrl: post.imageUrl,
        caption: post.finalText
      });

      post.status = "published";
      post.metaContainerId = result.creationId;
      post.metaPublishId = result.publishId;
      post.publishedAt = new Date().toISOString();
      post.updatedAt = new Date().toISOString();
      post.errorMessage = null;

      console.log(`Published post ${post.id}`);
    } catch (error) {
      post.status = "failed";
      post.errorMessage = JSON.stringify(error.response?.data || error.message);
      post.updatedAt = new Date().toISOString();

      console.error(`Failed post ${post.id}:`, post.errorMessage);
    }
  }
});

app.listen(PORT, () => {
  console.log(`AutoFlow Backend running on port ${PORT}`);
});
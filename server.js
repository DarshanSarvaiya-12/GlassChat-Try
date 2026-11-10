import express from 'express';
import axios from 'axios';

const app = express();
app.use(express.json());

const PORT = process.env.PORT || 3000;
const VERIFY_TOKEN = process.env.VERIFY_TOKEN;
const WHATSAPP_TOKEN = process.env.WHATSAPP_TOKEN;
const PHONE_NUMBER_ID = process.env.PHONE_NUMBER_ID;
const AI_BASE_URL = process.env.AI_BASE_URL;
const AI_API_KEY = process.env.AI_API_KEY;
const AI_MODEL_NAME = process.env.AI_MODEL_NAME || 'meta-llama/Llama-3-8b-chat-hf';

const processedMessages = new Set();

// Why we use this: Handles Meta webhook verification handshake
app.get('/webhook', (req, res) => {
  const mode = req.query['hub.mode'];
  const token = req.query['hub.verify_token'];
  const challenge = req.query['hub.challenge'];

  if (mode === 'subscribe' && token === VERIFY_TOKEN) {
    return res.status(200).send(challenge);
  }
  return res.sendStatus(403);
});

// Why we use this: Receives WhatsApp messages and triggers AI reply
app.post('/webhook', async (req, res) => {
  res.sendStatus(200);

  const body = req.body;

  try {
    const entry = body.entry?.[0];
    const changes = entry?.changes?.[0]?.value;
    const messageObj = changes?.messages?.[0];

    if (messageObj && messageObj.type === 'text') {
      const messageId = messageObj.id;
      const from = messageObj.from;
      const userText = messageObj.text.body;

      if (processedMessages.has(messageId)) return;
      processedMessages.add(messageId);
      setTimeout(() => processedMessages.delete(messageId), 60000);

      const aiReply = await getAiResponse(userText);
      await sendWhatsAppMessage(from, aiReply);
    }
  } catch (error) {
    console.error('Webhook processing error:', error.message);
  }
});

// Why we use this: Queries your custom AI endpoint
async function getAiResponse(userPrompt) {
  try {
    const response = await axios.post(
      `${AI_BASE_URL.replace(/\/+$/, '')}/chat/completions`,
      {
        model: AI_MODEL_NAME,
        messages: [
          {
            role: 'system',
            content:
              'You are an assistant for a T-shirt store. Help customers with sizing, designs, pricing, and how to place orders. Keep responses short and friendly.'
          },
          { role: 'user', content: userPrompt }
        ],
        temperature: 0.7,
        max_tokens: 250
      },
      {
        headers: {
          Authorization: `Bearer ${AI_API_KEY}`,
          'Content-Type': 'application/json'
        }
      }
    );

    return response.data.choices[0].message.content.trim();
  } catch (err) {
    console.error('AI API error:', err.response?.data || err.message);
    return "Hello! We received your message. What size or style are you looking for?";
  }
}

// Why we use this: Dispatches reply to WhatsApp user via Meta Cloud API
async function sendWhatsAppMessage(to, textMessage) {
  try {
    await axios.post(
      `https://graph.facebook.com/v21.0/${PHONE_NUMBER_ID}/messages`,
      {
        messaging_product: 'whatsapp',
        recipient_type: 'individual',
        to: to,
        type: 'text',
        text: { body: textMessage }
      },
      {
        headers: {
          Authorization: `Bearer ${WHATSAPP_TOKEN}`,
          'Content-Type': 'application/json'
        }
      }
    );
  } catch (error) {
    console.error('WhatsApp send error:', error.response?.data || error.message);
  }
}

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

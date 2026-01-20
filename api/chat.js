export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    const { messages } = req.body;

    const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${process.env.OPENROUTER_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: "mistralai/mistral-7b-instruct",
        messages,
        temperature: 0.7,
        max_tokens: 180
      })
    });

    const data = await response.json();

    // 🔒 SAFETY CHECK
    if (!data || !data.choices || !data.choices.length) {
      return res.status(200).json({
        content:
          "Samajh raha hoon — thoda heavy lag raha hoga. Main yahin hoon, batao kya sabse zyada bother kar raha hai?"
      });
    }

    return res.status(200).json({
      content: data.choices[0].message.content
    });

  } catch (err) {
    console.error("API ERROR:", err);
    return res.status(200).json({
      content:
        "Lagta hai thodi technical problem aa gayi. Par koi baat nahi — hum baat continue kar sakte hain. Batao?"
    });
  }
}

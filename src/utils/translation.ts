export async function multilingualTranslate(text: string, targetLang: string): Promise<string> {
    console.log("calling translation api");
    const response = await fetch("http://localhost:8081/translate", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ text, targetLang })
  });

  if (!response.ok) {
    throw new Error("Translation failed");
  }

  const data = await response.json();
  return data.translatedText;
}
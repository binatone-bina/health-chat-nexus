import { useState, useRef } from "react";
import { Button } from "@/components/ui/button";
import { Bot, Send, Mic, MicOff, ArrowRight, Clock, Brain } from "lucide-react";
import { useToast } from "@/hooks/use-toast";

interface Message {
  id: number;
  text: string;
  sender: "user" | "bot";
  timestamp: Date;
  isProcessing?: boolean;
  suggestions?: string[];
  doctorReferral?: {
    condition: string;
    specialtyNeeded: string;
  };
}

const MedicalChatbot = () => {
  const { toast } = useToast();
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      text: "Hi there! Please describe your symptoms.",
      sender: "bot",
      timestamp: new Date(),
      suggestions: [
        "I have a headache",
        "My throat is sore",
        "I have a fever",
        "My skin has a rash"
      ]
    }
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunks = useRef<Blob[]>([]);

  // Start/stop audio recording
  const handleMicClick = async () => {
    if (!isRecording) {
      // Start recording
      setAudioUrl(null);
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mediaRecorder = new MediaRecorder(stream);
        mediaRecorderRef.current = mediaRecorder;
        audioChunks.current = [];
        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunks.current.push(event.data);
          }
        };
        mediaRecorder.onstop = async () => {
  const audioBlob = new Blob(audioChunks.current, { type: "audio/webm" });

  // Prepare form data
  const formData = new FormData();
  formData.append("audio", audioBlob, "recording.webm");

  try {
    const response = await fetch("http://localhost:8081/record", {
      method: "POST",
      body: formData,
    });

    if (response.ok) {
      toast({
        title: "Upload complete",
        description: "Audio uploaded to the server.",
      });
      setAudioUrl(null); // Optionally clear the download link
    } else {
      toast({
        title: "Upload failed",
        description: "Could not upload audio to the server.",
        variant: "destructive",
      });
    }
  } catch (err) {
    toast({
      title: "Upload error",
      description: "An error occurred while uploading.",
      variant: "destructive",
    });
  }
};
        mediaRecorder.start();
        setIsRecording(true);
        toast({
          title: "Recording started",
          description: "Speak clearly to record your message."
        });
      } catch (err) {
        toast({
          title: "Microphone Error",
          description: "Could not access microphone.",
          variant: "destructive"
        });
      }
    } else {
      // Stop recording
      mediaRecorderRef.current?.stop();
      setIsRecording(false);
    }
  };

  const handleSendMessage = () => {
    if (input.trim() === "") return;

    const newMessage: Message = {
      id: messages.length + 1,
      text: input,
      sender: "user",
      timestamp: new Date()
    };

    setMessages(prev => [...prev, newMessage]);
    setInput("");
  };

  const currentLanguage = localStorage.getItem("language") || "en";
  console.log(currentLanguage);

  return (
    <div className="flex flex-col h-[calc(100vh-250px)] overflow-hidden rounded-lg shadow-md bg-white">
      <div className="bg-health-600 text-white p-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Bot className="h-5 w-5" />
          <h2 className="font-semibold">AI Health Assistant</h2>
        </div>
        <div className="text-xs bg-health-500 px-2 py-1 rounded-full">
          Powered by Medical AI
        </div>
      </div>
      
      <div className="flex-1 p-4 overflow-y-auto">
        <div className="space-y-4">
          {messages.map((message) => (
            <div 
              key={message.id} 
              className={`flex ${message.sender === "user" ? "justify-end" : "justify-start"}`}
            >
              <div 
                className={`
                  max-w-[80%] rounded-lg p-3
                  ${message.sender === "user" 
                    ? "bg-health-100 text-gray-800" 
                    : "bg-white border border-gray-200 shadow-sm"}
                `}
              >
                <div className="flex items-start gap-2">
                  {message.sender === "bot" && (
                    <Bot className="h-5 w-5 mt-1 text-health-600" />
                  )}
                  <div className="flex-1">
                    <div className="flex justify-between items-start">
                      <div className="text-sm font-medium">
                        {message.sender === "user" ? "You" : "AI Assistant"}
                      </div>
                      <div className="text-xs text-gray-400 ml-2">
                        {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </div>
                    </div>
                    <p className="mt-1 text-sm">{message.text}</p>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
      
      <div className="border-t p-3">
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            className={`rounded-full ${isRecording ? 'bg-red-100 text-red-500' : ''}`}
            onClick={handleMicClick}
          >
            {isRecording ? <MicOff className="h-5 w-5" /> : <Mic className="h-5 w-5" />}
          </Button>
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
            placeholder="Describe your symptoms..."
            className="flex-1 bg-gray-100 rounded-full px-4 py-2 focus:outline-none focus:ring-2 focus:ring-health-500"
          />
          <Button 
            size="sm"
            className="rounded-full"
            onClick={handleSendMessage}
            disabled={loading || input.trim() === ''}
          >
            <Send className="h-4 w-4" />
          </Button>
        </div>
        {/* Show download link for webm if available */}
        {audioUrl && (
          <div className="mt-2 px-2 text-xs text-gray-500">
            <a href={audioUrl} download="recording.webm" className="text-blue-600 underline">
              Download your recording (webm)
            </a>
          </div>
        )}
        <div className="mt-2 px-2">
          <p className="text-xs text-gray-400 flex items-center">
            <Brain className="h-3 w-3 mr-1" />
            AI assistant provides general information only. Always consult with a healthcare provider for medical advice.
          </p>
        </div>
      </div>
    </div>
  );
};

export default MedicalChatbot;
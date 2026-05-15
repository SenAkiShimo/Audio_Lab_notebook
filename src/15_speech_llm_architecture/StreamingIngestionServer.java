import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class StreamingIngestionServer {

    // Thread pool dedicated to streaming audio chunk ingestion
    private final ExecutorService ingestionThreadPool = Executors.newFixedThreadPool(4);

    // Context sliding window buffer maintained on the server side
    private final List<float[]> streamingContextWindow = new ArrayList<>();
    private final int maxWindowSize = 10;

    /**
     * Ingests variable-length audio feature chunks from clients asynchronously.
     * @param audioChunk 100ms audio feature array
     * @return CompletableFuture handle for async tracking
     */
    public CompletableFuture<Void> ingestAudioChunkStreamAsync(float[] audioChunk) {
        return CompletableFuture.runAsync(() -> {
            String threadName = Thread.currentThread().getName();
            
            synchronized (streamingContextWindow) {
                // 1. Sliding window flow control
                // Remove the oldest frame if the window has reached maximum size
                if (streamingContextWindow.size() >= maxWindowSize) {
                    streamingContextWindow.remove(0);
                }
                
                // 2. Append the incoming acoustic feature chunk to the end of the window
                streamingContextWindow.add(audioChunk);
                System.out.printf("[%s] [INGESTION] Captured 100ms audio slice | Window size: %d/%d \n", 
                    threadName, streamingContextWindow.size(), maxWindowSize);
            }
            
            // 3. Simulate asynchronous delivery and processing latency
            try {
                Thread.sleep(50); // Simulate 50ms network I/O or downstream calculation delay
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }, ingestionThreadPool);
    }

    // Shut down thread pool resources
    public void shutdownServer() {
        ingestionThreadPool.shutdown();
    }

    public static void main(String[] args) throws Exception {
        System.out.println("============================================================");
        System.out.println("[RUN] Activating Java enterprise streaming audio ingestion server test...");
        
        StreamingIngestionServer server = new StreamingIngestionServer();
        List<CompletableFuture<Void>> streamingPipelineTasks = new ArrayList<>();
        
        // Simulate client sending 12 audio chunks continuously over a live connection
        for (int i = 0; i < 12; ++i) {
            float[] mockAudioChunk100ms = new float[80]; // 80-dimensional feature vector per frame
            mockAudioChunk100ms[0] = (float) Math.random();
            
            CompletableFuture<Void> task = server.ingestAudioChunkStreamAsync(mockAudioChunk100ms);
            streamingPipelineTasks.add(task);
            
            Thread.sleep(30); // Interval between streaming packets from the client
        }
        
        // Wait for all asynchronous tasks to complete
        CompletableFuture.allOf(streamingPipelineTasks.toArray(new CompletableFuture[0])).get();
        server.shutdownServer();
        
        System.out.println("[SUCCESS] Concurrent streaming audio ingestion test complete.");
        System.out.println("============================================================");
    }
}

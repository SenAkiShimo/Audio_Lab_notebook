import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class SecurityGateway {
    private static final int MAX_ALLOWABLE_DURATION_SECONDS = 30;
    private static final int REQUIRED_BIT_DEPTH = 16;
    private static final int ALLOWED_SAMPLING_RATE_16K = 16000;
    private static final int WAV_HEADER_SIZE = 44;
    
    // 16,000 Hz * (16 bits / 8) = 32,000 bytes/second. The maximum theoretical size for 30 seconds is 960,000 bytes, plus a 44-byte header.
    private static final long MAX_ALLOWABLE_BYTES = ((long) ALLOWED_SAMPLING_RATE_16K * (REQUIRED_BIT_DEPTH / 8) * MAX_ALLOWABLE_DURATION_SECONDS) + WAV_HEADER_SIZE;

    private final ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors() * 2);

    public CompletableFuture<Boolean> validateStreamAsync(byte[] rawByteStream) {
        // Pass in a custom thread pool executor.
        return CompletableFuture.supplyAsync(() -> {
            // Key Optimization 3: Fail-Fast. If the array length directly exceeds the maximum possible size, immediately reject the request and forgo any subsequent calculations.
            if (rawByteStream == null || rawByteStream.length < WAV_HEADER_SIZE || rawByteStream.length > MAX_ALLOWABLE_BYTES) {
                return false;
            }

            ByteBuffer headerReader = ByteBuffer.wrap(rawByteStream, 0, WAV_HEADER_SIZE).order(ByteOrder.LITTLE_ENDIAN);
            int sampleRate = headerReader.getInt(24);
            short bitDepth = headerReader.getShort(34);

            if (sampleRate != ALLOWED_SAMPLING_RATE_16K || bitDepth != REQUIRED_BIT_DEPTH) {
                return false;
            }

            // Key Optimization 4: Eliminate Magic Numbers and Prevent Integer Overflow. Use long to ensure precise calculation of large file lengths.
            long totalPcmBytes = (long) rawByteStream.length - WAV_HEADER_SIZE;
            long bytesPerSecond = (long) sampleRate * (bitDepth / 8);
            
            if (bytesPerSecond == 0) return false; // Strictly prevent division by zero
            
            double computedDuration = (double) totalPcmBytes / (double) bytesPerSecond;
            return computedDuration <= MAX_ALLOWABLE_DURATION_SECONDS;
        }, executor);
    }

    // Methods for Explicitly Shutting Down a Thread Pool
    public void shutdown() {
        executor.shutdown();
    }

    public static void main(String[] args) throws Exception {
        System.out.println("============================================================");
        System.out.println("[RUN]");
        SecurityGateway gateway = new SecurityGateway();

        // Craft a malicious byte stream that simulates a DoS attack by exceeding the maximum allowable duration, while still having a valid header to bypass initial checks.
        byte[] malicious_dos_stream = new byte[32000 * 50 + WAV_HEADER_SIZE];
        ByteBuffer buf = ByteBuffer.wrap(malicious_dos_stream).order(ByteOrder.LITTLE_ENDIAN);
        buf.putInt(24, 16000); 
        buf.putShort(34, (short) 16); 

        // Route to gateway
        CompletableFuture<Boolean> result = gateway.validateStreamAsync(malicious_dos_stream);

        // Block and wait
        boolean pass = result.get();
        System.out.println("[GATEWAY VERDICT] Validation complete. Approval status: " + (pass ? "Approved (PASS)" : "Blocked and tripped the connection (BLOCKED)"));
        System.out.println("============================================================");
        
        gateway.shutdown();
    }
}

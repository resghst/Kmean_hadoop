import java.io.IOException;
import java.util.*;
import java.lang.*;
import java.util.Map.*;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.io.SequenceFile.Writer;
import org.apache.hadoop.io.SequenceFile.Reader;
import org.apache.hadoop.io.compress.GzipCodec;
import org.apache.hadoop.mapreduce.*;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.SequenceFileOutputFormat;

public class DaliPM25 {

	// private static final String BASE_PATH = "/Kmean/";
	// private static final String CEN_PATH = "/Kmean/center/center";
	// private static final String INPUT_PATH = "/Kmean/INPUT/IN";
	// private static final String DICT_PATH = "/Kmean/DICT/OUT";
        
	private static String BASE_PATH = "/user/hank/";
	private static String CEN_PATH = "center";
	private static String INPUT_PATH = "IN";
	private static String DICT_PATH = "OUT";
        
    public static class Map extends Mapper<LongWritable, Text, Text, Text> {
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            String[] tokens = line.split(",");
            int flag = 0;
            if (tokens[1].equals("大里"))
                flag += 1;
            if (tokens[2].equals("PM2.5"))
                flag += 1;

            for (int i=3; i<tokens.length; i++) {
                tokens[i] = tokens[i].replaceAll("[^0-9.]", "");
                if (tokens[i].equals(""))
                tokens[i] = "0";
            }
            if (flag == 2) {
                String ret = tokens[0];
                for (int i=1; i<tokens.length; i++)
                ret = ret + "," + tokens[i];
                context.write(new Text(ret), new Text(""));
            }
        }
    } 
        
    public static class Reduce extends Reducer<Text, Text, Text, Text> {
        public void reduce(Text key, Iterable<Text> values, Context context) 
            throws IOException, InterruptedException {
                context.write(key, new Text(""));
        }
    }
        
    public static class SelectKMap extends Mapper<LongWritable, Text, Text, Text> {
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            context.write(new Text("0"), new Text(line));
        }
    } 
        
    public static class SelectKReduce extends Reducer<Text, Text, Text, Text> {
        public void reduce(Text key, Iterable<Text> values, Context context) 
        throws IOException, InterruptedException {
            List<Text> valueList = new ArrayList<Text>();
            for(Text val:values) {
                valueList.add(new Text(val.toString()));
            }
            Random rand = new Random();
            int limitValue = 0, si = 0, added = 0, len = 0, vallen = valueList.size(); 
            String outStr="", verStr="";
            Text line = new Text("");
            Integer[] indexArr = new Integer[]{-1,-1,-1,-1};
            String[] iat = new String[1];
            while(limitValue<4){
                outStr="";
                si = rand.nextInt(vallen);
                added = 0;
                for(int i = 0; i < 4; i++) {
                    if(indexArr[i]==si) added = 1;
                }
                if(added == 0){
                    indexArr[limitValue++] = si;
                    len = 0 ;
                    for(int j=0; j<valueList.size(); j++){
                        len++;
                        if(len==si) {
                            line = valueList.get(j);
                            break;
                        }
                    }
                    iat = line.toString().split(",");
                    for(int i = 3; i<iat.length; i++){
                        outStr += iat[i];
                        if(i!=iat.length-1) outStr += ",";
                    }
                    context.write(new Text(Integer.toString(limitValue-1)), new Text(outStr));
                }
            }
        }
    }


    public static class KMeanMap extends Mapper<LongWritable, Text, Text, Text> {
		HashMap<Integer, Text> centers = new HashMap<Integer, Text>();
		HashMap<String, String> dictWords = new HashMap<String, String>();
		private IntWritable classCenter;

		public void setup(Context context) throws IOException,
				InterruptedException {
			Configuration conf = context.getConfiguration();
			FileSystem fs = FileSystem.get(conf);
            
			Path PART_CEN_PATH = new Path(conf.get("CEN_PATH"));

            SequenceFile.Reader reader = new SequenceFile.Reader(fs, PART_CEN_PATH, conf);
			Text key = new Text();
			Text value = new Text();
			while (reader.next(key, value)) {
                // context.write(new Text("LOG"), new Text(key.toString() + "------" + value.toString()));
				centers.put(Integer.parseInt(key.toString()), new Text(value));
			}
			reader.close();
			super.setup(context);
		}
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString();
            float[] dis =  new float[4];

            String[] tokens = line.split(",");
            float[] onerow =  new float[24];
            for(int j=0; j<24 ;j++) onerow[j] = Float.parseFloat(tokens[j+3]);
            float min = 1000000000;
            int min_index = -1;
            for(Entry<Integer, Text> Cendroid:centers.entrySet()){
                int keyIndex= Cendroid.getKey();
                String[] SCentrol= Cendroid.getValue().toString().split(",");
                for (int j = 0; j<24; j++)  {
                    dis[keyIndex] += Math.pow(Float.parseFloat(SCentrol[j])-onerow[j], 2);
                }
                dis[keyIndex] = (float) Math.sqrt(dis[keyIndex]);
                if(min>dis[keyIndex]) {
                    min = dis[keyIndex];
                    min_index = keyIndex;
                }
            }
            switch(min_index) {
                case 0:
                    context.write(new Text("0"), new Text(line));
                    break;
                case 1:
                    context.write(new Text("1"), new Text(line));
                    break;
                case 2:
                    context.write(new Text("2"), new Text(line));
                    break;
                case 3:
                    context.write(new Text("3"), new Text(line));
                    break;
                default:
                    break;
            }
        }
    } 
            
    public static class KMeanReduce extends Reducer<Text, Text, Text, Text> {
        private float[][] center =  new float[4][24];
        public void reduce(Text key, Iterable<Text> values, Context context) 
            throws IOException, InterruptedException {
            // for(Text val:values) context.write(key, val);
            // if((key.toString()).equals("LOG")){
            //     for(Text val:values) context.write(key, val);
            //     return ;
            // }

            List<Text> valueList = new ArrayList<Text>();
            for(Text val:values) {
                valueList.add(new Text(val.toString()));
            }
            float[] total =  new float[24];

            for(int i=0; i<valueList.size(); i++){
                String[] tokens = (valueList.get(i).toString()).split(",");
                for(int j=0; j<24 ;j++)
                    total[j] = total[j] + (Float.parseFloat(tokens[j+3])/valueList.size());
                context.write(new Text(key+","), new Text(valueList.get(i)));
            }
            int center_index = Integer.parseInt(key.toString());
            for (int i = 0; i<24 ; i++)center[center_index][i] =  total[i];
            
        }

		public void cleanup(Context context) throws IOException,
				InterruptedException {
			Configuration conf = context.getConfiguration();
			FileSystem fs = FileSystem.get(conf);

            Text key = new Text();
            Text value = new Text();
            Path out = new Path( conf.get("NEXT_CEN_PATH") );
            // SequenceFile.Writer  writer = SequenceFile.createWriter(conf, Writer.file(out), 
            // Writer.keyClass(key.getClass()), Writer.valueClass(value.getClass()), 
            // Writer.compression(SequenceFile.CompressionType.BLOCK, new GzipCodec()));
            SequenceFile.Writer writer = null;
            SequenceFile.Writer.Option optPath = SequenceFile.Writer.file(out);
            SequenceFile.Writer.Option optKey = SequenceFile.Writer.keyClass(Text.class);
            SequenceFile.Writer.Option optVal = SequenceFile.Writer.valueClass(Text.class);
            writer = SequenceFile.createWriter(conf, optPath, optKey, optVal);
            
            String outStr = "";
            for(int i = 0; i<4; i++){
                outStr = "";
                for(int j = 0; j<24; j++){
                    outStr += Float.toString(center[i][j]);
                    if(j!=23) outStr += ",";
                }
                
                // context.write(new Text("LOG2"), new Text(Integer.toString(i) + "~~~" + outStr) );
                writer.append(new Text(Integer.toString(i)), new Text(outStr));
            }
            writer.close();
		}
    }

	
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
            
            Job job = new Job(conf, "DaliPM25");

		Path out = new Path(INPUT_PATH);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class); 

        //data is like a pipeline, input -> map -> reduce -> output
        //for every program, there will be only two functions, Map and Reduce
        job.setMapperClass(Map.class);
        job.setReducerClass(Reduce.class);
        job.setJarByClass(DaliPM25.class);

        //where the input from    
        job.setInputFormatClass(TextInputFormat.class);
        //where the output to
        job.setOutputFormatClass(TextOutputFormat.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, out);

        //when writing mapReduce program, you don't need to deal with I/O or execute
        //ONLY "map" and "reduce" function.

        job.waitForCompletion(true);

        //===============================================================================================

        Configuration confSelectK = new Configuration();
		Path in = new Path(INPUT_PATH);
		// out = new Path(CEN_PATH + "0");
        Job jobSelectK = new Job(confSelectK, "SelectK");

        jobSelectK.setMapperClass(SelectKMap.class);
        jobSelectK.setReducerClass(SelectKReduce.class);
        jobSelectK.setJarByClass(DaliPM25.class);

		FileSystem fs = FileSystem.get(confSelectK);

        Text key = new Text();
        Text value = new Text();
        out = new Path(BASE_PATH + CEN_PATH + "0");

		if (fs.exists(out) || fs.exists(in)) fs.delete(out, true);
        FileInputFormat.addInputPath(jobSelectK, in);
        FileOutputFormat.setOutputPath(jobSelectK, out);

        jobSelectK.setInputFormatClass(TextInputFormat.class);
        jobSelectK.setOutputFormatClass(SequenceFileOutputFormat.class);
        
        jobSelectK.setOutputKeyClass(Text.class);
        jobSelectK.setOutputValueClass(Text.class);

        jobSelectK.waitForCompletion(true);

        //===============================================================================================
        Configuration confKMean = new Configuration();
        Job jobKMean = new Job(confKMean, "KMean");
        for(int i = 0; i<30; i++){
            confKMean = new Configuration();
            in = new Path(INPUT_PATH);
            confKMean.set("CEN_PATH", BASE_PATH + CEN_PATH + Integer.toString(i) + "/part-r-00000");
            confKMean.set("NEXT_CEN_PATH", BASE_PATH + CEN_PATH + Integer.toString(i+1) + "/part-r-00000");
                        
            out = new Path(DICT_PATH+Integer.toString(i));
            jobKMean = new Job(confKMean, "KMean");

            jobKMean.setMapperClass(KMeanMap.class);
            jobKMean.setReducerClass(KMeanReduce.class);
            jobKMean.setJarByClass(DaliPM25.class);

            fs = FileSystem.get(confKMean);
           
            System.out.println("------------------------------------------");
            System.out.println("THIS ROUND: "+Integer.toString(i));
            SequenceFile.Reader reader = new SequenceFile.Reader(fs, new Path(BASE_PATH + CEN_PATH + Integer.toString(i) + "/part-r-00000"), conf);
			Text k = new Text();
			Text v = new Text();
            Float[] singleCentrol = new Float[24];
            String[] SCentrol = new String[1];
			while (reader.next(k, v)) {
                SCentrol = v.toString().split(",");
                System.out.println("|||| LOG"+ k.toString() + "------" + Arrays.toString(SCentrol));
                // for (int j = 0; j<24; j++)  {
                //     System.out.println(Float.toString(Float.parseFloat(SCentrol[j])));
                // }
			}
            reader.close();

            if (fs.exists(out) || fs.exists(in)) fs.delete(out, true);
            FileInputFormat.addInputPath(jobKMean, in);
            FileOutputFormat.setOutputPath(jobKMean, out);

            jobKMean.setInputFormatClass(TextInputFormat.class);
            jobKMean.setOutputFormatClass(TextOutputFormat.class);
            
            jobKMean.setOutputKeyClass(Text.class);
            jobKMean.setOutputValueClass(Text.class);

            jobKMean.waitForCompletion(true);
        }
    }
}

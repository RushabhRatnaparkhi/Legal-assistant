import mongoose from "mongoose";

const connectDB=async ()=>{
    try{
        const connectionInstance=await mongoose.connect(`${process.env.MONGODB_URI}/"Legal-AI-agent"`);
        console.log(`\n MongoDB connected !! DB Host: ${connectionInstance.connection.name}`);
        return {
            status:200,
            msg:"Successfully Connected"
        };
    }
    catch(error){
        console.error("Error connecting Mongodb: "+error);
        return error;
        // process.exit(1);
    }
}

export {connectDB};
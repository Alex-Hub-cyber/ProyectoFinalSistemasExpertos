const express = require("express");
const app=express();
app.get("/", (req, res)=>{
 //   res.send("Hola word")
 res.sendFile("C:/Users/ALEX/Documents/PISOS_ALQUILER_TF/app_pisos/Index.html")
});

app.listen(30001,()=>{
    console.log("server running on port", 30001);
});
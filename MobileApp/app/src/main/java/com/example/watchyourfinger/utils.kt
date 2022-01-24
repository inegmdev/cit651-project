package com.example.watchyourfinger

import androidx.compose.ui.graphics.Color
import com.example.watchyourfinger.ui.theme.*


fun getColor(prediction: String): Color {
    return when (prediction) {
        "toxic" -> toxicColor
        "severe_toxic" -> severe_toxicColor
        "obscene" -> obsceneColor
        "threat" -> threatColor
        "insult" -> insultColor
        "identity_hate" -> identity_hateColor


        else -> nonToxicColor
    }
}

fun getAbbrev(prediction: String): String {
    return when (prediction) {
        "toxic" -> "TX"
        "severe_toxic" -> "ST"
        "obscene" -> "O"
        "threat" -> "TH"
        "insult" -> "I"
        "identity_hate" -> "H"


        else -> "OK"
    }
}
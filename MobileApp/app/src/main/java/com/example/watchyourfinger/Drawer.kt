package com.example.watchyourfinger


import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material.MaterialTheme
import androidx.compose.material.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp


@Composable
fun Drawer(onToggleDrawer: () -> Unit ) {

    val predictions = listOf("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
    Column(
        modifier = Modifier.background(Color.White)//.width(10.dp)
    ) {


        predictions.forEach {
            Item(longText = it)
            Spacer(Modifier.height(8.dp))
        }



    }
}


@Composable
fun Item(  longText: String)
{


    Row(
        verticalAlignment = Alignment.CenterVertically,
        modifier = Modifier
            .fillMaxWidth()

            .padding(10.dp)
    ) {
        RoundTag(color = getColor(longText), shortText = getAbbrev(longText))
        Spacer(Modifier.width(16.dp))
        Text(text = longText, style = MaterialTheme.typography.h5)
    }
}
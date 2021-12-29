package com.example.watchyourfinger

//import androidx.compose.material.icons.outlined.Reorder
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.material.Icon
import androidx.compose.material.IconButton
import androidx.compose.material.Text
import androidx.compose.material.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign

@Composable
fun TopBarWithDrawer(title: String, onToggleDrawer:() -> Unit) {
    TopAppBar(
        title = {
            Text(
                text = title,
                modifier = Modifier.fillMaxWidth(),
                textAlign = TextAlign.Center
            )
        },
        navigationIcon = {
            IconButton(onClick = {
                onToggleDrawer()
            }) {
                Icon(
                    painter = painterResource(id = R.drawable.ic_outline_reorder_24) ,
                    contentDescription = "drawer",
                )
            }
        },

    )
}